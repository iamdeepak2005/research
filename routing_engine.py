#!/usr/bin/env python3
"""
routing_engine.py — Congestion-aware routing engine with Dijkstra's algorithm.

Maintains the switch-level topology graph.  When the predictor flags a
congested link, this module computes an alternate shortest path using
Dijkstra's algorithm (weighted by link utilization) and returns the
OpenFlow FlowMod instructions the controller should install.

Supports dynamic topology editing (add/remove switches, links) from the
web dashboard's topology editor.
"""

import heapq
from collections import deque
from utils import SWITCH_LINKS, HOST_SWITCH, IP_HOST


class RoutingEngine:
    """Graph-based alternate-path computation for the SDN topology using Dijkstra."""

    # Default port-to-neighbor mapping (4-switch topology).
    # Overridden if topo_config is passed to __init__.
    PORT_TO_NEIGHBOR = {
        ("s1", 1): "h1", ("s1", 2): "s2", ("s1", 3): "s3", ("s1", 4): "s4",
        ("s2", 1): "h2", ("s2", 2): "s1", ("s2", 3): "s3", ("s2", 4): "s4",
        ("s3", 1): "h3", ("s3", 2): "s1", ("s3", 3): "s2", ("s3", 4): "s4",
        ("s4", 1): "h4", ("s4", 2): "s1", ("s4", 3): "s2", ("s4", 4): "s3",
    }

    def __init__(self, topo_config=None):
        """
        Args:
            topo_config: optional dict from utils.generate_topology(n).
                         If None, uses the hardcoded 4-switch topology.
        """
        if topo_config:
            # Dynamic topology
            links = topo_config["switch_links"]
            self.PORT_TO_NEIGHBOR = dict(topo_config["port_to_neighbor"])
            self._host_switch = dict(topo_config["host_switch"])
            self._ip_host = dict(topo_config["ip_host"])
        else:
            links = SWITCH_LINKS
            self._host_switch = dict(HOST_SWITCH)
            self._ip_host = dict(IP_HOST)

        # Build adjacency list
        self.graph = {}
        for a, b in links:
            self.graph.setdefault(a, set()).add(b)
            self.graph.setdefault(b, set()).add(a)

        # Link weights: keyed by frozenset({sw1, sw2}) → float weight
        # Default weight 1.0 for all links (equal cost)
        self._link_weights = {}
        for a, b in links:
            self._link_weights[frozenset({a, b})] = 1.0

        # Store the switch_links set for serialization
        self._switch_links = set()
        for a, b in links:
            self._switch_links.add((a, b))

        print(f"[RoutingEngine] Graph (Dijkstra-enabled): {self.graph}")

    # ------------------------------------------------------------------
    # Dynamic topology editing
    # ------------------------------------------------------------------
    def add_switch(self, switch_id, host_id=None, host_ip=None):
        """
        Add a new switch (and optionally its host) to the topology.
        Returns True if added, False if already exists.
        """
        if switch_id in self.graph:
            return False
        self.graph[switch_id] = set()

        if host_id and host_ip:
            self._host_switch[host_id] = switch_id
            self._ip_host[host_ip] = host_id
            # Assign port 1 to host
            port = 1
            self.PORT_TO_NEIGHBOR[(switch_id, port)] = host_id

        print(f"[RoutingEngine] Added switch {switch_id}")
        return True

    def remove_switch(self, switch_id):
        """Remove a switch and all its links from the topology."""
        if switch_id not in self.graph:
            return False

        # Remove all links involving this switch
        neighbors = list(self.graph.get(switch_id, []))
        for nbr in neighbors:
            self.graph[nbr].discard(switch_id)
            link_key = frozenset({switch_id, nbr})
            self._link_weights.pop(link_key, None)
            self._switch_links.discard((switch_id, nbr))
            self._switch_links.discard((nbr, switch_id))

        del self.graph[switch_id]

        # Remove port mappings
        keys_to_remove = [k for k in self.PORT_TO_NEIGHBOR if k[0] == switch_id]
        for k in keys_to_remove:
            del self.PORT_TO_NEIGHBOR[k]

        # Remove host mapping
        hosts_to_remove = [h for h, s in self._host_switch.items() if s == switch_id]
        for h in hosts_to_remove:
            del self._host_switch[h]
            ips_to_remove = [ip for ip, host in self._ip_host.items() if host == h]
            for ip in ips_to_remove:
                del self._ip_host[ip]

        print(f"[RoutingEngine] Removed switch {switch_id}")
        return True

    def add_link(self, sw1, sw2, weight=1.0):
        """Add a link between two switches."""
        if sw1 not in self.graph or sw2 not in self.graph:
            return False
        self.graph[sw1].add(sw2)
        self.graph[sw2].add(sw1)
        link_key = frozenset({sw1, sw2})
        self._link_weights[link_key] = weight
        self._switch_links.add((sw1, sw2))

        # Update port mappings
        existing_ports_sw1 = [k[1] for k in self.PORT_TO_NEIGHBOR if k[0] == sw1]
        next_port_sw1 = max(existing_ports_sw1) + 1 if existing_ports_sw1 else 2
        self.PORT_TO_NEIGHBOR[(sw1, next_port_sw1)] = sw2

        existing_ports_sw2 = [k[1] for k in self.PORT_TO_NEIGHBOR if k[0] == sw2]
        next_port_sw2 = max(existing_ports_sw2) + 1 if existing_ports_sw2 else 2
        self.PORT_TO_NEIGHBOR[(sw2, next_port_sw2)] = sw1

        print(f"[RoutingEngine] Added link {sw1} ↔ {sw2} (weight={weight})")
        return True

    def remove_link(self, sw1, sw2):
        """Remove a link between two switches."""
        self.graph.get(sw1, set()).discard(sw2)
        self.graph.get(sw2, set()).discard(sw1)
        link_key = frozenset({sw1, sw2})
        self._link_weights.pop(link_key, None)
        self._switch_links.discard((sw1, sw2))
        self._switch_links.discard((sw2, sw1))

        # Remove port mappings for this link
        keys_to_remove = []
        for k, v in self.PORT_TO_NEIGHBOR.items():
            if k[0] == sw1 and v == sw2:
                keys_to_remove.append(k)
            elif k[0] == sw2 and v == sw1:
                keys_to_remove.append(k)
        for k in keys_to_remove:
            del self.PORT_TO_NEIGHBOR[k]

        print(f"[RoutingEngine] Removed link {sw1} ↔ {sw2}")
        return True

    def update_link_weights(self, utilizations):
        """
        Update link weights from current utilization predictions.

        utilizations: dict of {(switch_id, port_no): utilization_fraction}

        Weight formula: weight = 1 + utilization * 10
        This makes Dijkstra prefer less-utilized links significantly.
        A link at 0% utilization has weight 1.0.
        A link at 100% utilization has weight 11.0.
        """
        for (sw, port), util in utilizations.items():
            neighbor = self.PORT_TO_NEIGHBOR.get((sw, port))
            if neighbor and neighbor.startswith("s"):
                link_key = frozenset({sw, neighbor})
                # Weight increases with utilization: highly congested = expensive
                self._link_weights[link_key] = 1.0 + util * 10.0

    def get_link_weight(self, sw1, sw2):
        """Get the current weight for a link between two switches."""
        return self._link_weights.get(frozenset({sw1, sw2}), 1.0)

    def get_topology_snapshot(self):
        """Return current topology as a serializable dict for the editor."""
        switches = sorted(self.graph.keys())
        links = []
        seen = set()
        for sw, neighbors in self.graph.items():
            for nbr in neighbors:
                link_key = frozenset({sw, nbr})
                if link_key not in seen:
                    seen.add(link_key)
                    links.append({
                        "from": sw, "to": nbr,
                        "weight": self._link_weights.get(link_key, 1.0)
                    })

        hosts = []
        for host_id, switch_id in self._host_switch.items():
            ip = next((ip for ip, h in self._ip_host.items() if h == host_id), "")
            hosts.append({"id": host_id, "switch": switch_id, "ip": ip})

        return {
            "switches": switches,
            "hosts": hosts,
            "links": links,
        }

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
    def dijkstra_shortest_path(self, src_sw, dst_sw, blocked_links=None):
        """
        Dijkstra's shortest path using link utilization as edge weights.

        blocked_links: set of frozenset({sw1, sw2}) pairs to avoid.
        Returns list of switch IDs or [] if unreachable.

        Uses a min-heap priority queue. Edge weights are based on current
        link utilization: weight = 1 + utilization * 10.
        This ensures the algorithm chooses the least-congested path,
        not just the shortest hop count.
        """
        blocked = blocked_links or set()

        # Min-heap: (cumulative_cost, node, path)
        heap = [(0.0, src_sw, [src_sw])]
        visited = set()

        while heap:
            cost, node, path = heapq.heappop(heap)

            if node == dst_sw:
                return path

            if node in visited:
                continue
            visited.add(node)

            for nbr in self.graph.get(node, []):
                link = frozenset({node, nbr})
                if nbr not in visited and link not in blocked:
                    edge_weight = self._link_weights.get(link, 1.0)
                    heapq.heappush(heap, (cost + edge_weight, nbr, path + [nbr]))

        return []

    # ------------------------------------------------------------------
    def shortest_path(self, src_sw, dst_sw, blocked_links=None):
        """
        Shortest path using Dijkstra's algorithm with utilization-weighted edges.

        blocked_links: set of frozenset({sw1, sw2}) pairs to avoid.
        Returns list of switch IDs or [] if unreachable.

        Note: This now uses Dijkstra instead of BFS. When all link weights
        are equal (no utilization data), it behaves identically to BFS.
        When utilization data is available, it picks the least-congested path.
        """
        return self.dijkstra_shortest_path(src_sw, dst_sw, blocked_links)

    # ------------------------------------------------------------------
    def compute_alternate_path(self, src_ip, dst_ip, congested_links,
                               utilizations=None):
        """
        Given source/dest IPs and a list of congested (switch, port) pairs,
        find an alternate path that avoids those links.

        congested_links: list of (switch_id, port_no)
        utilizations: optional dict of {(sw, port): util} for weight updates
        Returns: list of switch IDs for the new path, or [] if none found.
        """
        # Update weights from latest utilization data
        if utilizations:
            self.update_link_weights(utilizations)

        src_host = self._ip_host.get(src_ip)
        dst_host = self._ip_host.get(dst_ip)
        if not src_host or not dst_host:
            return []

        src_sw = self._host_switch[src_host]
        dst_sw = self._host_switch[dst_host]

        # Convert congested (switch, port) → blocked link frozensets
        blocked = set()
        for sw, port in congested_links:
            neighbor = self.PORT_TO_NEIGHBOR.get((sw, port))
            if neighbor and neighbor.startswith("s"):  # only block switch-switch links
                blocked.add(frozenset({sw, neighbor}))

        # Use Dijkstra to find the optimal path avoiding blocked links
        path = self.dijkstra_shortest_path(src_sw, dst_sw, blocked_links=blocked)
        if not path:
            # Fallback: try without blocking (best-effort, still weighted)
            path = self.dijkstra_shortest_path(src_sw, dst_sw)
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

    print("\n--- Dijkstra shortest path s1 → s4 (equal weights) ---")
    print("  ", engine.shortest_path("s1", "s4"))

    print("\n--- Dijkstra shortest path s1 → s4 (block s1-s4 direct) ---")
    blocked = {frozenset({"s1", "s4"})}
    print("  ", engine.shortest_path("s1", "s4", blocked_links=blocked))

    # Test with utilization weights
    print("\n--- Dijkstra with s1-s4 congested (high weight) ---")
    engine.update_link_weights({("s1", 4): 0.95})  # makes s1-s4 expensive
    print("  ", engine.shortest_path("s1", "s4"))
    print("  (Should prefer s1→s2→s4 or s1→s3→s4 over direct s1→s4)")

    print("\n--- Alternate path h1 → h4 with congested (s1, 4) ---")
    alt = engine.compute_alternate_path("10.0.0.1", "10.0.0.4", [("s1", 4)])
    print("  ", alt)
    print("   Actions:", engine.path_to_flow_actions(alt))

    # Test dynamic editing
    print("\n--- Dynamic topology editing ---")
    engine.add_switch("s5", "h5", "10.0.0.5")
    engine.add_link("s5", "s1")
    engine.add_link("s5", "s4")
    print("  After adding s5:", engine.graph)
    print("  Path s5 → s2:", engine.shortest_path("s5", "s2"))
    print("  Topology snapshot:", engine.get_topology_snapshot())
