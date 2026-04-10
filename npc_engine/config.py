"""
NPC Engine Configuration.

Extends PIE's config with NPC-specific settings:
  - World directory (where NPC profiles, state, examples live)
  - Social graph configuration
  - Gossip propagation rules
  - Trust ripple settings
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class GossipRules:
    max_hops: int = 2
    decay_per_hop: float = 0.5
    min_significance: float = 0.2
    propagation_delay: int = 1  # turns before gossip reaches next NPC


@dataclass
class TrustRippleConfig:
    enabled: bool = True
    positive_factor: float = 0.3
    negative_factor: float = 0.15
    max_ripple: int = 10


@dataclass
class NPCEngineConfig:
    """Configuration for the NPC Engine."""
    # World settings
    world_dir: str = "data/worlds/ashenvale"
    world_name: str = ""

    # PIE config path (for the underlying engine)
    pie_config: str = ""  # defaults to PIE's config.yaml

    # Paths derived from world_dir
    profiles_dir: str = ""  # auto-set from world_dir
    state_dir: str = ""     # auto-set from world_dir
    examples_dir: str = ""  # auto-set from world_dir
    quests_path: str = ""   # auto-set from world_dir
    world_yaml: str = ""    # auto-set from world_dir

    # Social system
    gossip: GossipRules = field(default_factory=GossipRules)
    trust_ripple: TrustRippleConfig = field(default_factory=TrustRippleConfig)

    # Starting NPC
    active_npc: str = ""

    def __post_init__(self):
        """Derive paths from world_dir."""
        world = Path(self.world_dir)
        if not self.profiles_dir:
            self.profiles_dir = str(world / "npc_profiles")
        if not self.state_dir:
            self.state_dir = str(world / "npc_state")
        if not self.examples_dir:
            self.examples_dir = str(world / "examples")
        if not self.quests_path:
            self.quests_path = str(world / "player_quests.yaml")
        if not self.world_yaml:
            self.world_yaml = str(world / "world.yaml")

    @classmethod
    def load(cls, config_path: str) -> "NPCEngineConfig":
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config not found at {path}, using defaults")
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        config = cls(
            world_dir=data.get("world_dir", "data/worlds/ashenvale"),
            world_name=data.get("world_name", ""),
            pie_config=data.get("pie_config", ""),
            active_npc=data.get("active_npc", ""),
        )

        # Gossip rules
        gossip_data = data.get("gossip_rules", {})
        if gossip_data:
            config.gossip = GossipRules(
                max_hops=gossip_data.get("max_hops", 2),
                decay_per_hop=gossip_data.get("decay_per_hop", 0.5),
                min_significance=gossip_data.get("min_significance", 0.2),
                propagation_delay=gossip_data.get("propagation_delay", 1),
            )

        # Trust ripple
        ripple_data = data.get("trust_ripple", {})
        if ripple_data:
            config.trust_ripple = TrustRippleConfig(
                enabled=ripple_data.get("enabled", True),
                positive_factor=ripple_data.get("positive_factor", 0.3),
                negative_factor=ripple_data.get("negative_factor", 0.15),
                max_ripple=ripple_data.get("max_ripple", 10),
            )

        return config
