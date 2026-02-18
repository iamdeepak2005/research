#!/usr/bin/env python3
"""
controller.py — Ryu SDN controller with telemetry collection, ML prediction,
and proactive congestion-aware rerouting.

Run:
    ryu-manager controller.py

pip install:
    pip install ryu eventlet scikit-learn joblib pandas numpy
"""

import time
import json

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import (
    CONFIG_DISPATCHER,
    MAIN_DISPATCHER,
    set_ev_cls,
)
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, arp
from ryu.lib import hub

from telemetry_collector import TelemetryCollector
from dataset_logger import DatasetLogger
from predictor import Predictor
from routing_engine import RoutingEngine
from utils import (
    POLL_INTERVAL,
    UTILIZATION_THRESHOLD,
    now_iso,
    bytes_to_utilization,
)


class PredictiveController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # MAC table: {dpid: {mac: port}}
        self.mac_to_port = {}

        # Port-number mapping: {dpid: {peer_dpid: port_no}}
        # Built dynamically from LLDP / link events or hard-coded below.
        self.switch_port_map = {}

        # Subsystems
        self.telemetry = TelemetryCollector()
        self.logger_csv = DatasetLogger()
        self.predictor = Predictor()
        self.routing = RoutingEngine()

        # Previous port stats for rate calculation
        self._prev_port_stats = {}

        # Start periodic stats polling
        self.monitor_thread = hub.spawn(self._monitor_loop)

    # ==================================================================
    #  Switch handshake — install table-miss flow
    # ==================================================================
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        ofproto = dp.ofproto
        parser = dp.ofproto_parser

        # Table-miss: send to controller
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(
            ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER
        )]
        self._add_flow(dp, 0, match, actions)
        self.logger.info(f"[Controller] Switch {dp.id} connected")

    # ==================================================================
    #  Utility — install a flow rule
    # ==================================================================
    def _add_flow(self, datapath, priority, match, actions, idle=0, hard=0):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(
            ofproto.OFPIT_APPLY_ACTIONS, actions
        )]
        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=priority,
            match=match,
            instructions=inst,
            idle_timeout=idle,
            hard_timeout=hard,
        )
        datapath.send_msg(mod)

    # ==================================================================
    #  Packet-In — basic L2 learning switch + forwarding
    # ==================================================================
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        dp = msg.datapath
        ofproto = dp.ofproto
        parser = dp.ofproto_parser
        in_port = msg.match["in_port"]

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        # Ignore LLDP
        if eth.ethertype == 0x88cc:
            return

        dst = eth.dst
        src = eth.src
        dpid = dp.id

        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        # Install a flow to avoid future packet-ins
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            self._add_flow(dp, 1, match, actions, idle=60)

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(
            datapath=dp,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=data,
        )
        dp.send_msg(out)

    # ==================================================================
    #  Periodic stats monitor
    # ==================================================================
    def _monitor_loop(self):
        while True:
            for dp in self._get_datapaths():
                self._request_port_stats(dp)
            hub.sleep(POLL_INTERVAL)

    def _get_datapaths(self):
        """Return list of connected datapaths."""
        try:
            from ryu.topology.api import get_all_switch
            switches = get_all_switch(self)
            return [sw.dp for sw in switches]
        except Exception:
            # Fallback: empty until switches connect
            return []

    def _request_port_stats(self, datapath):
        parser = datapath.ofproto_parser
        req = parser.OFPPortStatsRequest(datapath, 0, datapath.ofproto.OFPP_ANY)
        datapath.send_msg(req)

    # ==================================================================
    #  Port stats reply handler
    # ==================================================================
    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_reply_handler(self, ev):
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        ts = now_iso()

        for stat in body:
            port_no = stat.port_no
            # Skip local/controller ports
            if port_no > 1000:
                continue

            record = {
                "timestamp": ts,
                "switch_id": f"s{dpid}",
                "port_no": port_no,
                "tx_packets": stat.tx_packets,
                "rx_packets": stat.rx_packets,
                "tx_bytes": stat.tx_bytes,
                "rx_bytes": stat.rx_bytes,
                "duration_sec": stat.duration_sec if stat.duration_sec > 0 else 1,
            }

            # Store & log
            self.telemetry.add_record(record)
            self.logger_csv.log(record)

        # After processing all ports for this switch, run prediction
        self._run_prediction(ev.msg.datapath)

    # ==================================================================
    #  Prediction + rerouting
    # ==================================================================
    def _run_prediction(self, datapath):
        predictions, congested = self.predictor.predict_all(self.telemetry)

        if congested:
            self.logger.warning(
                f"[Predictor] Congestion predicted on: {congested}"
            )
            for sw, port in congested:
                pred_val = predictions.get((sw, port), 0)
                self.logger.warning(
                    f"  {sw} port {port} → predicted {pred_val:.2%}"
                )

            # Export latest telemetry snapshot
            self.telemetry.export_json()

            # Attempt reroute (example: h1 → h4)
            # In production you'd inspect the affected flows.
            alt_path = self.routing.compute_alternate_path(
                "10.0.0.1", "10.0.0.4", congested
            )
            if alt_path:
                self.logger.info(
                    f"[RoutingEngine] Reroute via: {' → '.join(alt_path)}"
                )
                self._install_reroute(datapath, alt_path)
        else:
            if predictions:
                self.logger.info(
                    f"[Predictor] All links OK — max predicted "
                    f"{max(predictions.values()):.2%}"
                )

    # ------------------------------------------------------------------
    def _install_reroute(self, datapath, path):
        """
        Install flow rules along the alternate path for traffic
        between h1 (10.0.0.1) and h4 (10.0.0.4).

        This is a simplified implementation that pushes a high-priority
        rule on the ingress switch to redirect to the next hop.
        """
        if len(path) < 2:
            return

        # We need the datapath objects for each switch in the path.
        # For now, install on the current datapath if it's the ingress.
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto

        # High-priority match for h1 → h4
        match = parser.OFPMatch(
            eth_type=0x0800,
            ipv4_src="10.0.0.1",
            ipv4_dst="10.0.0.4",
        )

        # Determine output port for next hop on this switch
        # This would use a proper port mapping in production.
        # For demo: use FLOOD as fallback, which still reaches the dest.
        actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]

        self._add_flow(datapath, 10, match, actions, idle=30)
        self.logger.info(
            f"[Reroute] Installed high-priority flow on switch {datapath.id}"
        )
