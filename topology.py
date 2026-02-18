#!/usr/bin/env python3
"""
topology.py — Mininet topology for SDN congestion avoidance research.

Topology (mesh with redundant paths):

    h1 --- s1 ------- s2 --- h2
            |  \   /  |
            |    X    |
            |  /   \  |
    h3 --- s3 ------- s4 --- h4

Links: s1-s2, s1-s3, s1-s4, s2-s3, s2-s4, s3-s4  (full mesh between switches)

Run:
    sudo python3 topology.py
"""

from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink


def create_topology():
    net = Mininet(
        controller=RemoteController,
        switch=OVSKernelSwitch,
        link=TCLink,
        autoSetMacs=True,
    )

    info("*** Adding controller (remote Ryu on 127.0.0.1:6633)\n")
    c0 = net.addController(
        "c0",
        controller=RemoteController,
        ip="127.0.0.1",
        port=6633,
    )

    info("*** Adding switches\n")
    s1 = net.addSwitch("s1", protocols="OpenFlow13")
    s2 = net.addSwitch("s2", protocols="OpenFlow13")
    s3 = net.addSwitch("s3", protocols="OpenFlow13")
    s4 = net.addSwitch("s4", protocols="OpenFlow13")

    info("*** Adding hosts\n")
    h1 = net.addHost("h1", ip="10.0.0.1/24")
    h2 = net.addHost("h2", ip="10.0.0.2/24")
    h3 = net.addHost("h3", ip="10.0.0.3/24")
    h4 = net.addHost("h4", ip="10.0.0.4/24")

    info("*** Adding host-switch links\n")
    net.addLink(h1, s1, bw=10)  # 10 Mbps
    net.addLink(h2, s2, bw=10)
    net.addLink(h3, s3, bw=10)
    net.addLink(h4, s4, bw=10)

    info("*** Adding switch-switch links (full mesh)\n")
    net.addLink(s1, s2, bw=10)  # s1-s2
    net.addLink(s1, s3, bw=10)  # s1-s3
    net.addLink(s1, s4, bw=10)  # s1-s4
    net.addLink(s2, s3, bw=10)  # s2-s3
    net.addLink(s2, s4, bw=10)  # s2-s4
    net.addLink(s3, s4, bw=10)  # s3-s4

    info("*** Starting network\n")
    net.start()

    info("*** Running CLI\n")
    CLI(net)

    info("*** Stopping network\n")
    net.stop()


if __name__ == "__main__":
    setLogLevel("info")
    create_topology()
