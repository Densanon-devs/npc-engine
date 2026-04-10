"""
Social Graph — Defines NPC-to-NPC connections and relationships.

Loaded from world.yaml, provides adjacency queries for gossip
propagation and trust ripple effects.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class NPCConnection:
    """A directed social connection between two NPCs."""
    from_id: str
    to_id: str
    relationship: str = "acquaintance"  # mentor, friend, rival, business, family, duty, acquaintance
    closeness: float = 0.5             # 0.0-1.0
    gossip_filter: str = "all"         # all, personal, trade, military, lore, quest, none


class SocialGraph:
    """
    Manages NPC-to-NPC social connections.
    Supports directed relationships (A→B may differ from B→A).
    """

    def __init__(self, world_yaml_path: str = ""):
        self.connections: list[NPCConnection] = []
        self._adjacency: dict[str, list[NPCConnection]] = defaultdict(list)

        if world_yaml_path:
            self._load(world_yaml_path)

    def _load(self, path: str) -> None:
        """Load social graph from world.yaml."""
        p = Path(path)
        if not p.exists():
            logger.debug(f"No world.yaml at {path}, social graph empty")
            return

        try:
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to load world.yaml: {e}")
            return

        graph_data = data.get("social_graph", {})
        for conn_data in graph_data.get("connections", []):
            conn = NPCConnection(
                from_id=conn_data.get("from", ""),
                to_id=conn_data.get("to", ""),
                relationship=conn_data.get("relationship", "acquaintance"),
                closeness=conn_data.get("closeness", 0.5),
                gossip_filter=conn_data.get("gossip_filter", "all"),
            )
            if conn.from_id and conn.to_id:
                self.connections.append(conn)
                self._adjacency[conn.from_id].append(conn)

        logger.info(f"Social graph loaded: {len(self.connections)} connections, "
                    f"{len(self._adjacency)} NPCs")

    def get_connections(self, npc_id: str) -> list[NPCConnection]:
        """Get all outgoing connections from an NPC."""
        return self._adjacency.get(npc_id, [])

    def get_connection(self, from_id: str, to_id: str) -> Optional[NPCConnection]:
        """Get a specific connection between two NPCs."""
        for conn in self._adjacency.get(from_id, []):
            if conn.to_id == to_id:
                return conn
        return None

    def get_reachable(self, npc_id: str, max_hops: int = 2) -> dict[str, int]:
        """Get all NPCs reachable from npc_id within max_hops. Returns {npc_id: hop_count}."""
        reachable: dict[str, int] = {}
        frontier = [npc_id]
        visited = {npc_id}

        for hop in range(1, max_hops + 1):
            next_frontier = []
            for current in frontier:
                for conn in self._adjacency.get(current, []):
                    if conn.to_id not in visited:
                        reachable[conn.to_id] = hop
                        visited.add(conn.to_id)
                        next_frontier.append(conn.to_id)
            frontier = next_frontier
            if not frontier:
                break

        return reachable

    def get_all_npcs(self) -> set[str]:
        """Get all NPC IDs that appear in the graph."""
        npcs = set()
        for conn in self.connections:
            npcs.add(conn.from_id)
            npcs.add(conn.to_id)
        return npcs

    def get_closeness(self, from_id: str, to_id: str) -> float:
        """Get closeness between two NPCs (0.0 if not connected)."""
        conn = self.get_connection(from_id, to_id)
        return conn.closeness if conn else 0.0

    def get_gossip_filter(self, from_id: str, to_id: str) -> str:
        """Get gossip filter for a connection."""
        conn = self.get_connection(from_id, to_id)
        return conn.gossip_filter if conn else "none"
