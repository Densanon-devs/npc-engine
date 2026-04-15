#!/usr/bin/env python3
"""
Synthetic NPC world generator for Story Director scaling tests.

Produces a brand-new world under ``data/worlds/<output>/`` with N towns of
M NPCs each, complete with profile YAMLs and a social graph that the
existing engine loader path consumes unchanged. Inter-town connections
form a sparse ring; intra-town connections are dense. Each NPC gets a
templated identity, four world facts grounded in the town's setting,
four personal facts (backstory + relationship + secret + motivation),
optionally one quest, and a minimal capability set (trust + emotional
state — heavier capabilities like knowledge_gate and goals are skipped
to keep capability state files small at scale).

The generator is deterministic given a seed. It is NOT meant to produce
narratively rich content — its job is to surface scaling bottlenecks in
the Director (snapshot O(N) growth, rotation thrash, arc cluster
selection, ledger latency, peak RSS). Run it once per test
configuration; the output directory is overwritten if it already
exists.

Usage:
    python generate_synthetic_world.py --towns 25 --npcs-per-town 20 --output synthetic_500
    python generate_synthetic_world.py --towns 5 --npcs-per-town 5 --output synthetic_25 --seed 7
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml

NPC_ROOT = Path(__file__).parent.resolve()


FIRST_NAMES = [
    "Aldric", "Bren", "Cormac", "Dara", "Edda", "Fenwick", "Gilda", "Halvar",
    "Iona", "Joren", "Kestrel", "Linnea", "Maerik", "Nessa", "Olwen", "Perrin",
    "Quill", "Rorik", "Sera", "Tobias", "Una", "Vance", "Wren", "Xanthe",
    "Yarrow", "Zephyr", "Aria", "Bram", "Cael", "Dorian", "Esme", "Faelan",
]

LAST_NAMES = [
    "Ashforth", "Brackwater", "Croft", "Dunmoor", "Elderwood", "Frosthold",
    "Greycote", "Hollybrook", "Ironvale", "Junipersworn", "Kettleberry",
    "Larksong", "Mossfield", "Nightwhisper", "Oakenshield", "Pendrake",
    "Quartzfen", "Riversend", "Stoneacre", "Thornbury",
]

ROLES = [
    ("blacksmith",    "the village forge",                 "iron rings, bellows hiss, soot in the rafters"),
    ("baker",         "the bakehouse off the square",      "the smell of yeast and the warm flagstones"),
    ("healer",        "the herb-garden cottage",           "rows of dried roots and a clay mortar"),
    ("scholar",       "the old library above the chapel",  "tall ladders and the dust of forgotten ledgers"),
    ("hunter",        "a low-roofed house at the wood's edge", "deer hides drying on a frame outside"),
    ("farmer",        "a stone cottage on the south road", "a kitchen garden and a leaning fence"),
    ("merchant",      "the trading post by the gate",      "stacked crates and a battered ledger"),
    ("guard",         "the watchhouse beside the gate",    "a worn longsword and a brass horn"),
    ("miner",         "a shack near the cliff path",       "a pickaxe and a smudged lantern"),
    ("sailor",        "the harborside boarding house",     "a coil of tarred rope and a brass spyglass"),
    ("scribe",        "a cramped office by the magistrate", "an inkwell and a stack of unfinished writs"),
    ("brewer",        "the brewhouse cellar",              "oak casks and the sweet tang of malt"),
]

PERSONALITIES = [
    "gruff but fair, slow to trust outsiders",
    "warm and gossipy, takes an interest in everyone",
    "weary and resigned, has seen too many bad winters",
    "ambitious and impatient, always looking for the next opportunity",
    "contemplative and quiet, prefers the company of books to people",
    "cautious and secretive, never says more than necessary",
    "bold and direct, will tell you exactly what they think",
    "pragmatic and hard-nosed, sees the world as a ledger of debts",
    "kindhearted and earnest, easily moved by other people's troubles",
    "suspicious and watchful, convinced that something is always wrong",
]

SPEECH_STYLES = [
    "short sentences, no flourishes",
    "wandering, peppered with proverbs from the old country",
    "blunt and impatient, doesn't suffer questions twice",
    "warm and digressive, always with a story to tell",
    "formal and precise, like reading from a book",
    "rough and clipped, the cadence of the dockside",
]

MOODS = ["calm", "weary", "cautious", "watchful", "contemplative", "resigned"]

TOWN_THEMES = [
    ("mining town",       "the deep tin mine that feeds three kingdoms"),
    ("farming village",   "the broad wheat fields that line the river"),
    ("port town",         "the deep harbor where the salt traders dock"),
    ("mountain pass",     "the only road through the spine of the range"),
    ("frontier outpost",  "the wooden palisade that holds back the wilderness"),
    ("river crossing",    "the old stone bridge that the tax collectors guard"),
    ("monastery hamlet",  "the bell tower that rings the canonical hours"),
    ("logging camp",      "the great pines that the timberers fell each spring"),
    ("salt flat",         "the white pans that the brine boilers harvest"),
    ("trading hub",       "the weekly market that draws caravans from four roads"),
]

TOWN_NAMES = [
    "Highmark", "Stonefall", "Greycote", "Ravenmoor", "Ashenrun", "Coldspring",
    "Brackwood", "Elmridge", "Foxhollow", "Glimwater", "Hookpoint", "Ironreach",
    "Jasperfield", "Kingsmoor", "Larkhaven", "Mistford", "Northwell", "Oakhurst",
    "Pinecrest", "Quartzfen", "Riverbend", "Saltmarch", "Thistledown", "Underhill",
    "Vexley", "Wolfsford", "Yarrowmere", "Ironbridge", "Cloudkeep", "Thornwell",
]

QUEST_TEMPLATES = [
    {
        "id_suffix":  "missing_tool",
        "name":       "The Missing {tool}",
        "description":"My {tool} has gone missing from the {workplace}. Find out who took it.",
        "reward":     "20 silver and my thanks",
        "objective":  "Search the {workplace} and ask around the town",
    },
    {
        "id_suffix":  "strange_noise",
        "name":       "Strange Noises at Night",
        "description":"For three nights running there has been a noise from the {landmark}. Investigate.",
        "reward":     "15 silver and a free meal",
        "objective":  "Visit the {landmark} after dark",
    },
    {
        "id_suffix":  "deliver_letter",
        "name":       "A Letter for the Next Town",
        "description":"Carry this sealed letter to my cousin in {neighbor_town}. Tell no one its contents.",
        "reward":     "30 silver",
        "objective":  "Deliver the letter to a contact in {neighbor_town}",
    },
    {
        "id_suffix":  "bandit_track",
        "name":       "The Bandits on the Road",
        "description":"There are tracks on the road north of the town. Find out where they go.",
        "reward":     "40 silver and a guard's recommendation",
        "objective":  "Follow the tracks past the watch line",
    },
]

TOOLS_BY_ROLE = {
    "blacksmith": "hammer", "baker": "kneading paddle", "healer": "mortar",
    "scholar": "writing quill", "hunter": "skinning knife", "farmer": "scythe",
    "merchant": "ledger", "guard": "polished horn", "miner": "lantern",
    "sailor": "spyglass", "scribe": "inkwell", "brewer": "tasting ladle",
}


def _slugify(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_").replace("'", "")


def _strip_article(s: str) -> str:
    """Strip a leading 'the '/'a '/'an ' so the result composes cleanly
    after another article in a template."""
    lower = s.lower()
    for prefix in ("the ", "an ", "a "):
        if lower.startswith(prefix):
            return s[len(prefix):]
    return s


def _make_npc(rng: random.Random, npc_id: str, first: str, last: str,
              town_name: str, town_theme: str, town_landmark: str,
              neighbor_town: str,
              other_npcs_in_town: list[tuple[str, str]]) -> dict:
    """
    Build one templated NPC profile dict. The id, first name, and last
    name are precomputed by the caller so the file id and in-profile
    name stay in sync. other_npcs_in_town is a list of (id, first_name)
    for neighbors in the same town so this NPC can name them in
    personal_knowledge / world_facts — that's what makes the social
    graph feel grounded instead of abstract.
    """
    name = f"{first} {last}"

    role, workplace, atmosphere = rng.choice(ROLES)
    personality = rng.choice(PERSONALITIES)
    speech = rng.choice(SPEECH_STYLES)
    mood = rng.choice(MOODS)
    age = rng.randint(22, 68)

    # Pick 2-3 other NPCs in the same town to mention by name. This is
    # the load-bearing thing that lets the Director's prompt actually
    # see local relationships at scale.
    sampled = rng.sample(other_npcs_in_town, min(3, len(other_npcs_in_town))) if other_npcs_in_town else []

    # World facts: 4 entries grounded in the town's specific setting.
    # Generic-but-named — the names are unique per world, so the LLM
    # can echo them back. workplace and atmosphere are stripped of
    # leading articles so they compose cleanly with the templates'
    # determiners.
    workplace_bare = _strip_article(workplace)
    atmosphere_bare = _strip_article(atmosphere)
    landmark_bare = _strip_article(town_landmark)
    world_facts = [
        f"{town_name} is a {town_theme} known for {town_landmark}",
        f"I work at the {workplace_bare}, surrounded by {atmosphere_bare}",
        f"The road from {town_name} runs to {neighbor_town} in two days' walking",
    ]
    if sampled:
        a_id, a_first = sampled[0]
        world_facts.append(
            f"{a_first} keeps a workshop on the same street as me"
        )
    else:
        world_facts.append(f"The watch posts a guard at the gate of {town_name} every evening")

    # Personal knowledge: backstory + relationship + secret + motivation.
    backstory_seed = rng.choice([
        f"was born in {neighbor_town} and moved to {town_name} as a child",
        f"learned the {role}'s trade from an uncle who died last winter",
        f"served two years in the watch before settling into the {role}'s work",
        f"has lived in this town for as long as anyone can remember",
        f"came from a logging camp far to the north and never speaks of why",
    ])
    if len(sampled) >= 2:
        b_id, b_first = sampled[1]
        relationship = (
            f"owes a debt of honor to {b_first} from an old quarrel over a stolen horse"
        )
    else:
        relationship = "has no living family in this town"
    secret = rng.choice([
        f"keeps a sealed packet of letters hidden under the floorboards of the {workplace_bare}",
        f"once paid a smuggler to carry a parcel from {neighbor_town} and has been waiting for the reply",
        f"saw something they should not have seen at the {landmark_bare} on a moonless night",
        f"has been quietly skimming a few coins from the {workplace_bare} for years",
        f"knows that the well at the edge of {town_name} runs to a place no map shows",
    ])
    motivation = rng.choice([
        f"wants to leave {town_name} for {neighbor_town} before the next harvest",
        f"wants to clear an old debt and finally rest easy",
        f"wants to see the bandit tracks on the north road dealt with for good",
        "wants to be respected the way their father was, no matter the cost",
        f"wants to know what happened to the people who left {town_name} last spring",
    ])
    personal_knowledge = [
        f"{backstory_seed}",
        relationship,
        secret,
        motivation,
    ]

    # Quest: only ~30% of NPCs ship one, otherwise focus rotation skips
    # most ticks and the Director's quest action gets starved.
    active_quests: list[dict] = []
    if rng.random() < 0.30:
        template = rng.choice(QUEST_TEMPLATES)
        tool = TOOLS_BY_ROLE.get(role, "tool")
        quest = {
            "id":          f"{npc_id}_{template['id_suffix']}",
            "name":        template["name"].format(tool=tool.title()),
            "description": template["description"].format(
                tool=tool, workplace=workplace_bare, landmark=landmark_bare,
                neighbor_town=neighbor_town,
            ),
            "status":      "available",
            "reward":      template["reward"],
            "objectives":  [template["objective"].format(
                workplace=workplace_bare, landmark=landmark_bare,
                neighbor_town=neighbor_town,
            )],
        }
        active_quests.append(quest)

    profile = {
        "identity": {
            "name":     name,
            "role":     role,
            "location": workplace,
            "world":    town_name,
            "age":      str(age),
            "personality": personality,
            "speech_style": speech,
        },
        "world_facts":         world_facts,
        "personal_knowledge":  personal_knowledge,
        "active_quests":       active_quests,
        "recent_events":       [],
        "capabilities": {
            "trust": {
                "enabled":       True,
                "initial_level": rng.randint(20, 50),
                "thresholds": {
                    "wary": 0, "neutral": 25, "friendly": 50, "trusted": 75,
                },
                "effects": {
                    "below_wary": "Speak evasively, withhold information",
                    "wary":       "Be cautious and guarded",
                    "neutral":    "Answer questions but do not volunteer secrets",
                    "friendly":   "Be warm and share useful information",
                    "trusted":    "Share secrets freely, offer better quest rewards",
                },
            },
            "emotional_state": {
                "enabled":      True,
                "baseline_mood": mood,
                "volatility":   round(rng.uniform(0.1, 0.4), 2),
                "decay_rate":   round(rng.uniform(0.05, 0.2), 2),
            },
        },
    }

    return profile


def _build_social_graph(town_layout: list[tuple[str, list[str]]],
                        rng: random.Random) -> list[dict]:
    """
    Two layers:
      1. Intra-town: each NPC connects to 3-5 random others in their
         own town. closeness 0.4-0.8.
      2. Inter-town: towns form a ring (i -> i-1, i -> i+1). Each pair
         of neighboring towns has 1-2 cross-town friendships.
         closeness 0.2-0.4.
    """
    connections: list[dict] = []
    relationships = ["friend", "family", "rival", "mentor", "business", "acquaintance"]
    gossip_filters = ["all", "personal", "trade", "military", "lore"]

    # Intra-town
    for town_name, npc_ids in town_layout:
        for npc_id in npc_ids:
            others = [n for n in npc_ids if n != npc_id]
            n_links = min(rng.randint(3, 5), len(others))
            for partner in rng.sample(others, n_links):
                connections.append({
                    "from":           npc_id,
                    "to":             partner,
                    "relationship":   rng.choice(relationships),
                    "closeness":      round(rng.uniform(0.4, 0.8), 2),
                    "gossip_filter":  rng.choice(gossip_filters),
                })

    # Inter-town (ring)
    n_towns = len(town_layout)
    for i, (town_name, npc_ids) in enumerate(town_layout):
        for delta in (-1, 1):
            j = (i + delta) % n_towns
            neighbor_town, neighbor_ids = town_layout[j]
            n_cross = rng.randint(1, 2)
            for _ in range(n_cross):
                a = rng.choice(npc_ids)
                b = rng.choice(neighbor_ids)
                connections.append({
                    "from":           a,
                    "to":             b,
                    "relationship":   rng.choice(["friend", "business", "acquaintance"]),
                    "closeness":      round(rng.uniform(0.2, 0.4), 2),
                    "gossip_filter":  rng.choice(gossip_filters),
                })

    return connections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--towns", type=int, default=25)
    parser.add_argument("--npcs-per-town", type=int, default=20)
    parser.add_argument("--output", type=str, default="synthetic_500",
                        help="World name; output goes to data/worlds/<output>/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    n_towns = args.towns
    n_per = args.npcs_per_town
    n_total = n_towns * n_per

    out_dir = NPC_ROOT / "data" / "worlds" / args.output
    if out_dir.exists():
        shutil.rmtree(out_dir)
    profiles_dir = out_dir / "npc_profiles"
    profiles_dir.mkdir(parents=True)
    # Per-world story pack directory. Even if we don't ship custom
    # lore/examples, creating this directory is what tells
    # StoryDirector._resolve_paths to isolate state/ledger/arcs into
    # the per-world location instead of polluting data/story_director/.
    story_dir = out_dir / "story"
    story_dir.mkdir(parents=True)

    print(f"Generating {n_towns} towns × {n_per} NPCs = {n_total} NPCs into {out_dir}")

    # Pick towns. If we ask for more towns than the static name pool,
    # extend with numeric suffixes so every town id is unique.
    if n_towns <= len(TOWN_NAMES):
        town_picks = rng.sample(TOWN_NAMES, n_towns)
    else:
        town_picks = list(TOWN_NAMES)
        for k in range(n_towns - len(TOWN_NAMES)):
            town_picks.append(f"Outpost{k+1:03d}")

    # Pass 1: precompute per-NPC (id, first, last) so the file id and
    # the in-profile name are guaranteed in sync, and so neighbors can
    # reference each other by first name. Town theme + landmark + neighbor
    # name are also picked here so pass 2 is purely templating.
    town_layout: list[dict] = []
    npc_index = 0
    for town_name in town_picks:
        theme, landmark = rng.choice(TOWN_THEMES)
        npcs_meta: list[tuple[str, str, str]] = []  # (id, first, last)
        for _ in range(n_per):
            first = rng.choice(FIRST_NAMES)
            last = rng.choice(LAST_NAMES)
            npc_id = f"{_slugify(first)}_{_slugify(last)}_{npc_index:03d}"
            npcs_meta.append((npc_id, first, last))
            npc_index += 1
        town_layout.append({
            "name":     town_name,
            "theme":    theme,
            "landmark": landmark,
            "npcs":     npcs_meta,
        })

    # Pass 2: build and write profiles. Each NPC sees its neighbors by
    # (id, first_name) so personal_knowledge can name them.
    for i, town in enumerate(town_layout):
        neighbor_name = town_layout[(i + 1) % len(town_layout)]["name"]
        other_pairs_full = [(nid, fname) for (nid, fname, _) in town["npcs"]]
        for (this_id, first, last) in town["npcs"]:
            others = [p for p in other_pairs_full if p[0] != this_id]
            profile = _make_npc(
                rng, this_id, first, last,
                town["name"], town["theme"], town["landmark"],
                neighbor_name, others,
            )
            profile_path = profiles_dir / f"{this_id}.yaml"
            with open(profile_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(profile, f, sort_keys=False, allow_unicode=True)

    # World file with social graph. The graph builder needs town -> id list,
    # so flatten the layout into the (name, [ids]) shape it expects.
    rng3 = random.Random(args.seed + 2)
    flat_layout = [(t["name"], [m[0] for m in t["npcs"]]) for t in town_layout]
    connections = _build_social_graph(flat_layout, rng3)

    world_yaml = {
        "world_name":  args.output.replace("_", " ").title(),
        "social_graph": {"connections": connections},
        "gossip_rules": {
            "max_hops":         2,
            "decay_per_hop":    0.5,
            "min_significance": 0.2,
            "propagation_delay": 1,
        },
        "trust_ripple": {
            "enabled":         True,
            "positive_factor": 0.3,
            "negative_factor": 0.15,
            "max_ripple":      10,
        },
    }
    with open(out_dir / "world.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(world_yaml, f, sort_keys=False, allow_unicode=True)

    # Minimal lore stub. The Director will use this whenever the
    # default lore would otherwise apply. Generic frontier setting so
    # the LLM has something to anchor on without leaking Ashenvale
    # specifics into the synthetic world.
    pretty_world_name = args.output.replace("_", " ").title()
    lore_path = story_dir / "lore.md"
    lore_path.write_text(
        f"# {pretty_world_name}\n\n"
        "A loose confederation of frontier towns at the far edge of an "
        "old kingdom. Each town keeps its own customs and grudges. "
        "Trade caravans link them along the river roads, and bandits "
        "shadow those same roads after dark. The crown has not sent a "
        "tax collector here in five years, and most of the towns have "
        "started to act as if they were never crown lands at all.\n\n"
        "## Standing tensions\n\n"
        "- The merchant houses in the larger towns squeeze the smaller "
        "ones with unfair freight rates.\n"
        "- Old debts and old grievances run deep — most townsfolk have "
        "memories long enough to remember the bad winter of '94.\n"
        "- Strangers are watched. People who arrive without an escort "
        "are assumed to be running from something.\n\n"
        "## Narrative rules\n\n"
        "- Every action should ground itself in the named town and "
        "named NPCs from the world snapshot. Generic 'a villager' or "
        "'someone' is wrong.\n"
        "- Quests should be small and concrete: a missing tool, a "
        "strange noise, a letter to deliver, tracks on the road.\n"
        "- Facts should reference real places (the town landmark, the "
        "river road, the watchhouse) and real people by name.\n",
        encoding="utf-8",
    )

    n_quests = sum(
        1
        for p in profiles_dir.glob("*.yaml")
        if (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("active_quests")
    )
    print(f"  wrote {n_total} profile YAMLs")
    print(f"  wrote {len(connections)} social connections")
    print(f"  ~{n_quests} NPCs ship with a starting quest "
          f"({100*n_quests/n_total:.0f}%)")
    print(f"  total profile size: {sum(p.stat().st_size for p in profiles_dir.glob('*.yaml'))/1024:.0f} KB")


if __name__ == "__main__":
    main()
