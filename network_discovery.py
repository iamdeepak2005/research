#!/usr/bin/env python3
"""
network_discovery.py — Real network topology discovery.

Discovers the actual network by:
  1. Reading local interface info (ipconfig)
  2. Scanning the ARP table for known devices
  3. Ping-sweeping the subnet for live hosts
  4. Running traceroute to find intermediate routers/hops
  5. Building a real topology graph from discovered devices

This is the SDN equivalent of LLDP flooding — we figure out what's
actually on the network instead of hardcoding "N switches".

Usage:
    from network_discovery import NetworkDiscovery
    disco = NetworkDiscovery()
    topo = disco.discover()  # returns topology config dict

Standalone test:
    python network_discovery.py
"""

import subprocess
import re
import socket
import ipaddress
import time
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
from utils import POLL_INTERVAL, LINK_CAPACITY_BYTES


# ── ANSI colors ──────────────────────────────────────────────────
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


class NetworkDiscovery:
    """
    Discovers real network topology using OS-level tools.
    No extra packages required — uses ipconfig, arp, ping, tracert.
    """

    def __init__(self, ping_timeout_ms=500, max_tracert_hops=8,
                 scan_subnet=True, max_scan_threads=50):
        self.ping_timeout_ms = ping_timeout_ms
        self.max_tracert_hops = max_tracert_hops
        self.scan_subnet = scan_subnet
        self.max_scan_threads = max_scan_threads

        # Discovered data
        self.local_interfaces = []    # list of {name, ip, subnet, gateway, mac}
        self.arp_entries = []          # list of {ip, mac, type, interface}
        self.live_hosts = []          # list of {ip, mac, hostname, role}
        self.routers = []             # list of {ip, hop_number, latency_ms}
        self.topology = None          # final topology config

    # ==================================================================
    #  MAIN DISCOVERY
    # ==================================================================
    def discover(self):
        """
        Run full network discovery. Returns a topology config dict
        compatible with utils.generate_topology().
        """
        print(f"\n{CYAN}{BOLD}╔══════════════════════════════════════════════════════════╗")
        print(f"║  NETWORK DISCOVERY — Scanning real network...           ║")
        print(f"╚══════════════════════════════════════════════════════════╝{RESET}\n")

        # Step 1: Local interfaces
        print(f"  {CYAN}[1/4] Reading network interfaces...{RESET}")
        self._discover_interfaces()
        for iface in self.local_interfaces:
            print(f"    {GREEN}✓{RESET} {iface['name']}: {iface['ip']}/{iface['subnet']} "
                  f"→ gateway {iface.get('gateway', 'none')}")

        # Step 2: ARP table
        print(f"\n  {CYAN}[2/4] Reading ARP table (known neighbors)...{RESET}")
        self._read_arp_table()
        dynamic_arps = [e for e in self.arp_entries if e['type'] == 'dynamic']
        print(f"    {GREEN}✓{RESET} Found {len(dynamic_arps)} dynamic ARP entries")

        # Step 3: Ping sweep
        if self.scan_subnet and self.local_interfaces:
            primary = self._get_primary_interface()
            if primary:
                print(f"\n  {CYAN}[3/4] Ping-sweeping {primary['network']} "
                      f"(discovering live hosts)...{RESET}")
                self._ping_sweep(primary)
                print(f"    {GREEN}✓{RESET} Found {len(self.live_hosts)} live devices")
        else:
            print(f"\n  {DIM}[3/4] Subnet scan disabled{RESET}")

        # Step 4: Traceroute
        print(f"\n  {CYAN}[4/4] Tracerouting to discover routers...{RESET}")
        self._discover_routers_tracert()
        print(f"    {GREEN}✓{RESET} Found {len(self.routers)} router hops")

        # Build topology
        print(f"\n  {CYAN}Building topology from discovered devices...{RESET}")
        topo = self._build_topology()
        self.topology = topo

        # Print summary
        self._print_summary(topo)

        return topo

    # ==================================================================
    #  STEP 1: Interface discovery
    # ==================================================================
    def _discover_interfaces(self):
        """Parse ipconfig to find active network interfaces."""
        try:
            result = subprocess.run(
                ["ipconfig", "/all"],
                capture_output=True, text=True, timeout=10
            )
            output = result.stdout
        except Exception:
            return

        self.local_interfaces = []
        current_iface = None

        for line in output.split("\n"):
            line = line.rstrip()

            # New adapter section
            adapter_match = re.match(r'^(\w[\w\s]+)\s+adapter\s+(.+?):', line)
            if adapter_match:
                if current_iface and current_iface.get('ip'):
                    self.local_interfaces.append(current_iface)
                current_iface = {
                    'type': adapter_match.group(1).strip(),
                    'name': adapter_match.group(2).strip(),
                    'ip': None, 'subnet': None, 'gateway': None,
                    'mac': None, 'network': None,
                }
                continue

            if not current_iface:
                continue

            # IPv4 address
            ip_match = re.search(r'IPv4 Address.*?:\s*([\d.]+)', line)
            if ip_match:
                current_iface['ip'] = ip_match.group(1)

            # Subnet mask
            sub_match = re.search(r'Subnet Mask.*?:\s*([\d.]+)', line)
            if sub_match:
                current_iface['subnet'] = sub_match.group(1)

            # Default gateway (IPv4 only)
            gw_match = re.search(r'Default Gateway.*?:\s*([\d.]+)', line)
            if gw_match:
                current_iface['gateway'] = gw_match.group(1)

            # MAC address
            mac_match = re.search(r'Physical Address.*?:\s*([\w-]+)', line)
            if mac_match:
                current_iface['mac'] = mac_match.group(1)

        # Don't forget the last one
        if current_iface and current_iface.get('ip'):
            self.local_interfaces.append(current_iface)

        # Compute network for each interface
        for iface in self.local_interfaces:
            if iface['ip'] and iface['subnet']:
                try:
                    net = ipaddress.IPv4Network(
                        f"{iface['ip']}/{iface['subnet']}", strict=False
                    )
                    iface['network'] = str(net)
                except ValueError:
                    pass

    def _get_primary_interface(self):
        """Get the primary interface (one with a gateway)."""
        for iface in self.local_interfaces:
            if iface.get('gateway'):
                return iface
        # Fallback: first interface with an IP
        return self.local_interfaces[0] if self.local_interfaces else None

    # ==================================================================
    #  STEP 2: ARP table
    # ==================================================================
    def _read_arp_table(self):
        """Read the system ARP table."""
        try:
            result = subprocess.run(
                ["arp", "-a"],
                capture_output=True, text=True, timeout=10
            )
        except Exception:
            return

        self.arp_entries = []
        current_iface_ip = None

        for line in result.stdout.split("\n"):
            line = line.strip()

            # Interface header
            iface_match = re.match(r'Interface:\s*([\d.]+)', line)
            if iface_match:
                current_iface_ip = iface_match.group(1)
                continue

            # ARP entry
            entry_match = re.match(
                r'([\d.]+)\s+([\w-]+)\s+(dynamic|static)', line
            )
            if entry_match:
                ip = entry_match.group(1)
                mac = entry_match.group(2)
                etype = entry_match.group(3)

                # Skip broadcast/multicast
                if ip.endswith('.255') or ip.startswith('224.') or ip.startswith('239.'):
                    continue
                if mac == 'ff-ff-ff-ff-ff-ff':
                    continue

                self.arp_entries.append({
                    'ip': ip,
                    'mac': mac,
                    'type': etype,
                    'interface': current_iface_ip,
                })

    # ==================================================================
    #  STEP 3: Ping sweep
    # ==================================================================
    def _ping_one(self, ip_str):
        """Ping a single IP. Returns (ip, alive, latency_ms)."""
        try:
            result = subprocess.run(
                ["ping", "-n", "1", "-w", str(self.ping_timeout_ms), ip_str],
                capture_output=True, text=True, timeout=3,
            )
            alive = result.returncode == 0 and "TTL=" in result.stdout
            latency = None
            if alive:
                lat_match = re.search(r'time[<=](\d+)', result.stdout)
                if lat_match:
                    latency = int(lat_match.group(1))
            return (ip_str, alive, latency)
        except Exception:
            return (ip_str, False, None)

    def _ping_sweep(self, iface):
        """Ping all IPs in the interface's subnet."""
        if not iface.get('network'):
            return

        try:
            network = ipaddress.IPv4Network(iface['network'], strict=False)
        except ValueError:
            return

        # Get all usable host IPs (skip network and broadcast)
        all_hosts = list(network.hosts())

        # Cap at 254 hosts (typical /24)
        if len(all_hosts) > 254:
            all_hosts = all_hosts[:254]

        # Already known hosts from ARP
        known_ips = {e['ip'] for e in self.arp_entries if e['type'] == 'dynamic'}

        # Build initial live_hosts from ARP
        self.live_hosts = []
        for entry in self.arp_entries:
            if entry['type'] == 'dynamic':
                hostname = self._resolve_hostname(entry['ip'])
                self.live_hosts.append({
                    'ip': entry['ip'],
                    'mac': entry['mac'],
                    'hostname': hostname,
                    'role': 'gateway' if entry['ip'] == iface.get('gateway') else 'host',
                    'latency_ms': None,
                })

        # Add self
        self.live_hosts.insert(0, {
            'ip': iface['ip'],
            'mac': iface.get('mac', 'self'),
            'hostname': socket.gethostname(),
            'role': 'self',
            'latency_ms': 0,
        })

        # Ping unknown IPs
        scan_ips = [str(ip) for ip in all_hosts
                    if str(ip) not in known_ips and str(ip) != iface['ip']]

        if not scan_ips:
            return

        print(f"    {DIM}Scanning {len(scan_ips)} unknown IPs "
              f"(parallelism: {self.max_scan_threads})...{RESET}", end="", flush=True)

        found = 0
        with ThreadPoolExecutor(max_workers=self.max_scan_threads) as pool:
            futures = {pool.submit(self._ping_one, ip): ip for ip in scan_ips}
            for future in as_completed(futures):
                ip, alive, latency = future.result()
                if alive:
                    found += 1
                    # Look up MAC from ARP (may have been added by the ping)
                    mac = self._lookup_mac(ip)
                    hostname = self._resolve_hostname(ip)
                    self.live_hosts.append({
                        'ip': ip,
                        'mac': mac or 'unknown',
                        'hostname': hostname,
                        'role': 'host',
                        'latency_ms': latency,
                    })

        print(f" found {found} more")

        # Re-read ARP after pinging (new entries may have appeared)
        self._read_arp_table()
        # Update MACs for hosts that were missing
        for host in self.live_hosts:
            if host['mac'] == 'unknown':
                mac = self._lookup_mac(host['ip'])
                if mac:
                    host['mac'] = mac

    def _lookup_mac(self, ip):
        """Look up MAC address from ARP entries."""
        for entry in self.arp_entries:
            if entry['ip'] == ip and entry['type'] == 'dynamic':
                return entry['mac']
        return None

    def _resolve_hostname(self, ip):
        """Try to resolve hostname for an IP."""
        try:
            hostname, _, _ = socket.gethostbyaddr(ip)
            return hostname
        except (socket.herror, socket.gaierror, OSError):
            return None

    # ==================================================================
    #  STEP 4: Traceroute
    # ==================================================================
    def _discover_routers_tracert(self):
        """Run traceroute to discover intermediate routers."""
        # Traceroute to a well-known IP
        targets = ["8.8.8.8"]
        self.routers = []

        for target in targets:
            try:
                result = subprocess.run(
                    ["tracert", "-d", "-h", str(self.max_tracert_hops),
                     "-w", "1000", target],
                    capture_output=True, text=True, timeout=30,
                )
            except Exception:
                continue

            for line in result.stdout.split("\n"):
                # Parse tracert output: "  1     2 ms    21 ms     3 ms  192.168.1.1"
                hop_match = re.match(
                    r'\s*(\d+)\s+(?:(\d+)\s+ms|[*])\s+(?:(\d+)\s+ms|[*])\s+'
                    r'(?:(\d+)\s+ms|[*])\s+([\d.]+)',
                    line
                )
                if hop_match:
                    hop_num = int(hop_match.group(1))
                    latencies = [hop_match.group(i) for i in (2, 3, 4)
                                 if hop_match.group(i)]
                    avg_lat = sum(int(l) for l in latencies) / len(latencies) if latencies else None
                    ip = hop_match.group(5)

                    # Don't add if it's our gateway (already found)
                    primary = self._get_primary_interface()
                    if primary and ip == primary.get('gateway'):
                        continue

                    # Don't duplicate
                    if not any(r['ip'] == ip for r in self.routers):
                        hostname = self._resolve_hostname(ip)
                        self.routers.append({
                            'ip': ip,
                            'hop_number': hop_num,
                            'latency_ms': avg_lat,
                            'hostname': hostname,
                        })

    # ==================================================================
    #  STEP 5: Build topology
    # ==================================================================
    def _build_topology(self):
        """
        Build SDN-compatible topology from discovered devices.

        Network model:
          - Routers/gateways → switches (switching/routing devices)
          - End devices → hosts (connected to their nearest switch)
          - Links are determined by actual network structure

        This is NOT a full-mesh assumption. The links are based on
        what we actually discovered:
          - Hosts connect to their gateway (switch)
          - Gateways connect to upstream routers (switch-to-switch links)
          - Traceroute hops form a chain of switches
        """
        primary = self._get_primary_interface()
        if not primary:
            # Fallback: return minimal topology
            return self._fallback_topology()

        # ── Identify switches (routers/gateways) ──
        switches = []  # list of {id, ip, hostname, role}

        # Gateway is always switch #1
        gw_ip = primary.get('gateway')
        if gw_ip:
            gw_host = self._find_host(gw_ip)
            switches.append({
                'id': 's1',
                'ip': gw_ip,
                'hostname': gw_host['hostname'] if gw_host else self._resolve_hostname(gw_ip),
                'mac': gw_host['mac'] if gw_host else self._lookup_mac(gw_ip),
                'role': 'gateway',
            })

        # Traceroute hops are additional switches
        for router in sorted(self.routers, key=lambda r: r['hop_number']):
            sw_id = f"s{len(switches) + 1}"
            switches.append({
                'id': sw_id,
                'ip': router['ip'],
                'hostname': router.get('hostname'),
                'mac': None,
                'role': 'isp_router',
                'latency_ms': router.get('latency_ms'),
            })

        if not switches:
            return self._fallback_topology()

        # ── Identify hosts (end devices) ──
        hosts = []  # list of {id, ip, hostname, mac, connected_to}

        # Self (our machine) is always host #1
        hosts.append({
            'id': 'h1',
            'ip': primary['ip'],
            'hostname': socket.gethostname(),
            'mac': primary.get('mac', 'self'),
            'connected_to': 's1',  # connected to gateway
        })

        # Other devices on the LAN are hosts connected to the gateway
        for device in self.live_hosts:
            if device['role'] == 'self':
                continue
            if device['role'] == 'gateway':
                continue
            # Skip if this IP is already a switch
            if any(s['ip'] == device['ip'] for s in switches):
                continue
            h_id = f"h{len(hosts) + 1}"
            hosts.append({
                'id': h_id,
                'ip': device['ip'],
                'hostname': device.get('hostname'),
                'mac': device.get('mac'),
                'connected_to': 's1',  # all LAN hosts connect to gateway
            })

        # ── Build links ──
        # Switch-to-switch links (gateway → ISP routers chain)
        switch_links = set()
        for i in range(len(switches) - 1):
            switch_links.add((switches[i]['id'], switches[i + 1]['id']))

        # If we have enough switches, also check for lateral connections
        # (In a real enterprise network, there might be redundant paths)
        # For now, we build a chain from the traceroute, plus any
        # extra connectivity we can infer.

        # ── Build host-switch mapping ──
        host_switch = {}
        for h in hosts:
            host_switch[h['id']] = h['connected_to']

        # ── Build IP mapping ──
        ip_host = {}
        for h in hosts:
            ip_host[h['ip']] = h['id']

        # ── Build port-to-neighbor mapping ──
        n_switches = len(switches)
        port_to_neighbor = {}

        for i, sw in enumerate(switches):
            port = 1
            # First: hosts connected to this switch
            connected_hosts = [h for h in hosts if h['connected_to'] == sw['id']]
            for h in connected_hosts:
                port_to_neighbor[(sw['id'], port)] = h['id']
                port += 1

            # Then: other switches connected to this one
            for j, other_sw in enumerate(switches):
                if i == j:
                    continue
                link_a = (sw['id'], other_sw['id'])
                link_b = (other_sw['id'], sw['id'])
                if link_a in switch_links or link_b in switch_links:
                    port_to_neighbor[(sw['id'], port)] = other_sw['id']
                    port += 1

        # ── Build switches_ports list ──
        switches_ports = list(port_to_neighbor.keys())

        # ── Device info for dashboard ──
        device_info = {
            'switches': [{
                'id': s['id'], 'ip': s['ip'],
                'hostname': s.get('hostname'),
                'role': s['role'],
                'mac': s.get('mac'),
            } for s in switches],
            'hosts': [{
                'id': h['id'], 'ip': h['ip'],
                'hostname': h.get('hostname'),
                'mac': h.get('mac'),
                'connected_to': h['connected_to'],
            } for h in hosts],
        }

        return {
            "switches": [s['id'] for s in switches],
            "hosts": [h['id'] for h in hosts],
            "switch_links": switch_links,
            "host_switch": host_switch,
            "ip_host": ip_host,
            "port_to_neighbor": port_to_neighbor,
            "switches_ports": switches_ports,
            "n_switches": n_switches,
            "device_info": device_info,
            "is_discovered": True,
        }

    def _find_host(self, ip):
        """Find a host entry by IP."""
        for h in self.live_hosts:
            if h['ip'] == ip:
                return h
        return None

    def _fallback_topology(self):
        """Minimal fallback topology if discovery fails."""
        from utils import generate_topology
        topo = generate_topology(2)
        topo['device_info'] = {
            'switches': [{'id': 's1', 'ip': 'unknown', 'hostname': 'router', 'role': 'gateway', 'mac': None}],
            'hosts': [{'id': 'h1', 'ip': 'self', 'hostname': socket.gethostname(),
                        'mac': None, 'connected_to': 's1'}],
        }
        topo['is_discovered'] = False
        return topo

    # ==================================================================
    #  DISPLAY
    # ==================================================================
    def _print_summary(self, topo):
        """Print a visual summary of the discovered topology."""
        di = topo.get('device_info', {})
        switches = di.get('switches', [])
        hosts = di.get('hosts', [])

        print(f"\n{CYAN}{BOLD}  ╔═══ DISCOVERED TOPOLOGY ═══════════════════════════════╗{RESET}")
        print(f"  {BOLD}  Switches (routers): {len(switches)}{RESET}")
        for s in switches:
            name = s.get('hostname') or s.get('role', '')
            print(f"    {CYAN}▸ {s['id'].upper()}{RESET} "
                  f"→ {s['ip']} ({name})"
                  f"{' [GATEWAY]' if s['role'] == 'gateway' else ''}"
                  f"{' [ISP]' if s['role'] == 'isp_router' else ''}")

        print(f"\n  {BOLD}  Hosts (end devices): {len(hosts)}{RESET}")
        for h in hosts:
            name = h.get('hostname') or ''
            link = h.get('connected_to', '?')
            print(f"    {GREEN}▸ {h['id'].upper()}{RESET} "
                  f"→ {h['ip']} ({name}) "
                  f"⟶ {link.upper()}")

        print(f"\n  {BOLD}  Links: {len(topo['switch_links'])}{RESET}")
        for a, b in topo['switch_links']:
            a_info = next((s for s in switches if s['id'] == a), {})
            b_info = next((s for s in switches if s['id'] == b), {})
            print(f"    {YELLOW}─{RESET} {a.upper()} ({a_info.get('ip','?')}) "
                  f"↔ {b.upper()} ({b_info.get('ip','?')})")

        print(f"{CYAN}{BOLD}  ╚══════════════════════════════════════════════════════╝{RESET}\n")

        # Visual ASCII topology
        print(f"  {DIM}Network map:{RESET}")
        if len(switches) >= 1:
            # Build ASCII art
            gw = switches[0]
            host_strs = [f"{h['id']}({h['ip']})" for h in hosts if h['connected_to'] == gw['id']]

            print(f"  {DIM}  ┌─────────────────────────────┐{RESET}")
            print(f"  {DIM}  │   Internet / ISP             │{RESET}")
            print(f"  {DIM}  └──────────┬──────────────────┘{RESET}")

            for sw in reversed(switches[1:]):
                label = f"{sw['id'].upper()} ({sw['ip']})"
                print(f"  {DIM}             │{RESET}")
                print(f"         {YELLOW}[{label}]{RESET}  ← ISP Router")

            print(f"  {DIM}             │{RESET}")
            gw_label = f"{gw['id'].upper()} ({gw['ip']})"
            print(f"         {CYAN}[{gw_label}]{RESET}  ← Your Gateway")

            for h in hosts:
                if h['connected_to'] == gw['id']:
                    marker = " ← YOU" if h['id'] == 'h1' else ""
                    label = f"{h['id'].upper()} ({h['ip']})"
                    hname = h.get('hostname') or ''
                    print(f"  {DIM}          ├──{RESET} {GREEN}[{label}]{RESET} {hname}{marker}")

        print()


# ══════════════════════════════════════════════════════════════
#  Standalone test
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    disco = NetworkDiscovery(ping_timeout_ms=300, max_tracert_hops=6)
    topo = disco.discover()

    print(f"\n{'═' * 50}")
    print("Topology config keys:")
    for k, v in topo.items():
        if isinstance(v, (set, list)):
            print(f"  {k}: ({len(v)} items)")
        elif isinstance(v, dict):
            print(f"  {k}: ({len(v)} entries)")
        else:
            print(f"  {k}: {v}")
