#!/usr/bin/env python3
"""
routing_engine.py — Congestion-aware routing engine.

Maintains the switch-level topology graph.  When the predictor flags a
congested link, this module computes an alternate shortest path (BFS)
and returns the OpenFlow FlowMod instructions the controller should install.
"""

from collections import deque
from utils import SWITCH_LINKS, HOST_SWITCH, IP_HOST


class RoutingEngine:
    """Graph-based alternate-path computation for the SDN topology."""

    # Port-to-neighbor mapping derived from topology.py link order.
    # In Mininet, port numbers are assigned sequentially per switch.
    # Host links are port 1, then inter-switch links follow.
    #
    # topology.py link order:
    #   h1-s1 (s1:1), h2-s2 (s2:1), h3-s3 (s3:1), h4-s4 (s4:1)
    #   s1-s2 (s1:2, s2:2)
    #   s1-s3 (s1:3, s3:2)
    #   s1-s4 (s1:4, s4:2)
    #   s2-s3 (s2:3, s3:3)
    #   s2-s4 (s2:4, s4:3)
    #   s3-s4 (s3:4, s4:4)
    PORT_TO_NEIGHBOR = {
        ("s1", 1): "h1", ("s1", 2): "s2", ("s1", 3): "s3", ("s1", 4): "s4",
        ("s2", 1): "h2", ("s2", 2): "s1", ("s2", 3): "s3", ("s2", 4): "s4",
        ("s3", 1): "h3", ("s3", 2): "s1", ("s3", 3): "s2", ("s3", 4): "s4",
        ("s4", 1): "h4", ("s4", 2): "s1", ("s4", 3): "s2", ("s4", 4): "s3",
    }

    def __init__(self):
        # Build adjacency list from the static mesh definition
        self.graph = {}  # switch → set of neighbour switches
        for a, b in SWITCH_LINKS:
            self.graph.setdefault(a, set()).add(b)
            self.graph.setdefault(b, set()).add(a)
        print(f"[RoutingEngine] Graph: {self.graph}")

    # ------------------------------------------------------------------
    def all_paths(self, src_sw, dst_sw, max_depth=6):
        """Return all simple paths (list of switch IDs) between two switches."""
        paths = []
        stack = [(src_sw, [src_sw])]
        while stack:
            node, path = stack.pop()
            if node == dst_sw:
                paths.append(path)
                continue
            if len(path) > max_depth:
                continue
            for nbr in self.graph.get(node, []):
                if nbr not in path:
                    stack.append((nbr, path + [nbr]))
        # Sort by length (prefer shorter)
        paths.sort(key=len)
        return paths

    # ------------------------------------------------------------------
    def shortest_path(self, src_sw, dst_sw, blocked_links=None):
        """
        BFS shortest path avoiding *blocked_links*.

        blocked_links: set of frozenset({sw1, sw2}) pairs to avoid.
        Returns list of switch IDs or [] if unreachable.
        """
        blocked = blocked_links or set()
        visited = {src_sw}
        queue = deque([(src_sw, [src_sw])])
        while queue:
            node, path = queue.popleft()
            if node == dst_sw:
                return path
            for nbr in self.graph.get(node, []):
                link = frozenset({node, nbr})
                if nbr not in visited and link not in blocked:
                    visited.add(nbr)
                    queue.append((nbr, path + [nbr]))
        return []

    # ------------------------------------------------------------------
    def compute_alternate_path(self, src_ip, dst_ip, congested_links):
        """
        Given source/dest IPs and a list of congested (switch, port) pairs,
        find an alternate path that avoids those links.

        congested_links: list of (switch_id, port_no)
        Returns: list of switch IDs for the new path, or [] if none found.
        """
        src_host = IP_HOST.get(src_ip)
        dst_host = IP_HOST.get(dst_ip)
        if not src_host or not dst_host:
            return []

        src_sw = HOST_SWITCH[src_host]
        dst_sw = HOST_SWITCH[dst_host]

        # Convert congested (switch, port) → blocked link frozensets
        blocked = set()
        for sw, port in congested_links:
            neighbor = self.PORT_TO_NEIGHBOR.get((sw, port))
            if neighbor and neighbor.startswith("s"):  # only block switch-switch links
                blocked.add(frozenset({sw, neighbor}))

        path = self.shortest_path(src_sw, dst_sw, blocked_links=blocked)
        if not path:
            # Fallback: try without blocking (best-effort)
            path = self.shortest_path(src_sw, dst_sw)
        return path

    # ------------------------------------------------------------------
    @staticmethod
    def path_to_flow_actions(path):
        """
        Convert a switch path [s1, s3, s4] into a list of
        (switch_id, out_port_hint) tuples.

        The controller maps these to actual port numbers using its
        switch-port tables.
        """
        actions = []
        for i, sw in enumerate(path):
            next_sw = path[i + 1] if i + 1 < len(path) else None
            actions.append({"switch": sw, "next_hop": next_sw})
        return actions


# ------------------------------------------------------------------
# Quick standalone test
# ------------------------------------------------------------------
if __name__ == "__main__":
    engine = RoutingEngine()

    print("\n--- All paths s1 → s4 ---")
    for p in engine.all_paths("s1", "s4"):
        print("  ", " → ".join(p))

    print("\n--- Shortest path s1 → s4 (no blocks) ---")
    print("  ", engine.shortest_path("s1", "s4"))

    print("\n--- Shortest path s1 → s4 (block s1-s4 direct) ---")
    blocked = {frozenset({"s1", "s4"})}
    print("  ", engine.shortest_path("s1", "s4", blocked_links=blocked))

    print("\n--- Alternate path h1 → h4 with congested (s1, 4) ---")
    alt = engine.compute_alternate_path("10.0.0.1", "10.0.0.4", [("s1", 4)])
    print("  ", alt)
    print("   Actions:", engine.path_to_flow_actions(alt))
