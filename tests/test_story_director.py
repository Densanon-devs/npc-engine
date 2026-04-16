#!/usr/bin/env python3
"""
Story Director tests.

Two modes:
  1. Offline unit tests (always run) — stub the engine + LLM, drive
     StoryDirector with crafted responses, verify parsing + dispatch.
  2. Integration smoke test (optional) — boots the real NPCEngine against
     Ashenvale, calls tick() once, asserts the world mutated. Skipped if
     the PIE model file is not present.

Usage:
    python tests/test_story_director.py
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

NPC_ROOT = Path(__file__).parent.parent.resolve()
PIE_ROOT = (NPC_ROOT.parent / "plug-in-intelligence-engine").resolve()

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NPC_ENGINE_DEV_MODE", "1")
logging.basicConfig(level=logging.WARNING)
for n in ["httpx", "huggingface_hub", "sentence_transformers", "faiss",
          "tqdm", "llama_cpp", "engine.npc_knowledge", "engine.npc_capabilities"]:
    logging.getLogger(n).setLevel(logging.ERROR)

sys.path.insert(0, str(NPC_ROOT))
sys.path.insert(0, str(PIE_ROOT))

from npc_engine.knowledge import NPCKnowledge, Quest  # noqa: E402
from npc_engine.story_director import (  # noqa: E402
    StoryDirector, FactLedger, ContradictionChecker,
    ArcPlanner, NarrativeArc,
)


# ── Offline stub engine ─────────────────────────────────────────

class _StubNPC:
    """Tiny stand-in for NPCKnowledge that supports add_quest + fields the
    snapshot builder pokes at."""

    def __init__(self, npc_id: str, role: str,
                 zone: str = "global", mobile: bool = False):
        self.identity = {"name": npc_id.title(), "role": role}
        self.world_facts: list[str] = []
        self.personal_knowledge: list[str] = []
        # Mirror the production dynamic lanes that engine.add_knowledge
        # writes to (see NPCKnowledge in npc_engine/knowledge.py).
        self.dynamic_world_facts: list[str] = []
        self.dynamic_personal_knowledge: list[str] = []
        self.quests: list[Quest] = []
        self.events: list[SimpleNamespace] = []
        # Zone fields mirror production NPCKnowledge (Phase 1 zone layer).
        # Default "global" keeps pre-zone stubs behaving as they did.
        self.home_zone = zone
        self.current_zone = zone
        self.mobile = mobile

    def add_quest(self, quest: Quest):
        self.quests.append(quest)

    def inject_event(self, description: str, source: str = "world"):
        self.events.append(SimpleNamespace(description=description, source=source))


class _StubKnowledgeManager:
    def __init__(self, profiles: dict):
        self.profiles = profiles

    def get(self, npc_id: str):
        return self.profiles.get(npc_id)

    def inject_event(self, npc_id: str, description: str):
        npc = self.profiles.get(npc_id)
        if npc:
            npc.inject_event(description, source="targeted")

    def inject_global_event(self, description: str):
        for npc in self.profiles.values():
            npc.inject_event(description, source="world")


class _StubPlayerQuests:
    def __init__(self):
        self.active_quests: list[dict] = []
        self.completed_quests: list[dict] = []


class _StubBaseModel:
    """Records prompts it receives and replays a scripted response list."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if not self._responses:
            return '{"action": "noop", "reason": "stub_exhausted"}'
        return self._responses.pop(0)


class _StubGoalsCap:
    """Mimic the shape of GoalsCapability.goals (priority-sorted list of dicts)."""

    def __init__(self, goals: list[dict]):
        # Priority-descending, same as the real capability does on init
        self.goals = sorted(goals, key=lambda g: g.get("priority", 0), reverse=True)


class _StubCapabilityManager:
    """Mimic the shape NPC capability managers expose — a .capabilities dict
    keyed by capability name. Tests attach _StubGoalsCap instances here so
    _peek_npc_goals / _build_focus_npc_bio can read them."""

    def __init__(self, capabilities: Optional[dict] = None):
        self.capabilities: dict = dict(capabilities or {})


class _StubPIE:
    def __init__(self, profiles):
        self.npc_knowledge = _StubKnowledgeManager(profiles)
        self.player_quests = _StubPlayerQuests()
        self.capability_managers: dict = {}
        self.base_model = None  # set per test


class _StubEngine:
    def __init__(self, profiles, world_name="Ashenvale"):
        self.pie = _StubPIE(profiles)
        self.config = SimpleNamespace(world_name=world_name)
        self.calls: list[tuple] = []  # for assertions in tests

    # Mirror the NPCEngine methods the director calls
    def inject_event(self, description: str, npc_id=None):
        if npc_id:
            self.pie.npc_knowledge.inject_event(npc_id, description)
        else:
            self.pie.npc_knowledge.inject_global_event(description)

    def add_knowledge(self, npc_id: str, fact: str, fact_type: str = "world") -> dict:
        npc = self.pie.npc_knowledge.get(npc_id)
        if not npc:
            return {"error": f"NPC '{npc_id}' not found"}
        if fact_type == "personal":
            npc.dynamic_personal_knowledge.append(fact)
        else:
            npc.dynamic_world_facts.append(fact)
        return {"npc_id": npc_id, "added": fact, "type": fact_type}

    def adjust_trust(self, npc_id: str, delta: int, reason: str = "") -> dict:
        self.calls.append(("adjust_trust", npc_id, delta, reason))
        return {"npc_id": npc_id, "delta": delta}

    def complete_quest(self, quest_id: str) -> dict:
        self.calls.append(("complete_quest", quest_id))
        # Mark on the player_quests stub so the snapshot reflects it
        self.pie.player_quests.completed_quests.append({"id": quest_id, "name": quest_id})
        return {"completed": quest_id}

    def accept_quest(self, quest_id: str, quest_name: str, given_by: str) -> dict:
        self.calls.append(("accept_quest", quest_id, quest_name, given_by))
        self.pie.player_quests.active_quests.append({
            "id": quest_id, "name": quest_name, "given_by": given_by, "status": "active",
        })
        return {"accepted": quest_id, "given_by": given_by}

    def add_profile(self, profile_path: str,
                     social_connections: list = None) -> dict:
        """Stub: load a YAML file and register the NPC in profiles."""
        from npc_engine.knowledge import NPCKnowledge
        p = Path(profile_path)
        if not p.exists():
            return {"ok": False, "reason": f"file not found: {p}"}
        npc_id = p.stem
        if npc_id in self.pie.npc_knowledge.profiles:
            return {"ok": False, "reason": f"already exists: {npc_id}"}
        npc = NPCKnowledge(str(p))
        self.pie.npc_knowledge.profiles[npc_id] = npc
        self.calls.append(("add_profile", npc_id))
        return {"ok": True, "npc_id": npc_id, "profile_path": str(p)}


_DEFAULT_STUB_PROFILES = (
    ("kael", "blacksmith"),
    ("bess", "innkeeper"),
    ("noah", "elder"),
)

_LARGE_STUB_PROFILES = (
    ("kael", "blacksmith"),
    ("bess", "innkeeper"),
    ("noah", "elder"),
    ("mara", "merchant"),
    ("tam", "hunter"),
    ("rin", "scribe"),
    ("eld", "priest"),
)


def _make_stub_engine(responses=None, profile_specs=None):
    """Build a stub engine. ``profile_specs`` lets callers override the
    default 3-NPC Ashenvale-ish cast; pass a sequence of ``(id, role)``
    or ``(id, role, zone)`` or ``(id, role, zone, mobile)`` tuples to
    exercise small-cast caps, multi-action plans above the
    _SMALL_CAST_THRESHOLD cutoff, or zoned worlds for the Phase 1
    zone layer tests."""
    specs = profile_specs or _DEFAULT_STUB_PROFILES
    profiles: dict = {}
    for spec in specs:
        if len(spec) == 2:
            npc_id, role = spec
            profiles[npc_id] = _StubNPC(npc_id, role)
        elif len(spec) == 3:
            npc_id, role, zone = spec
            profiles[npc_id] = _StubNPC(npc_id, role, zone=zone)
        elif len(spec) == 4:
            npc_id, role, zone, mobile = spec
            profiles[npc_id] = _StubNPC(npc_id, role, zone=zone, mobile=mobile)
        else:
            raise ValueError(f"bad profile spec: {spec}")
    engine = _StubEngine(profiles)
    engine.pie.base_model = _StubBaseModel(responses or [])
    return engine


def _isolate_state_file(tag: str):
    """Redirect StoryDirector's STATE_FILE, LEDGER_FILE, and ARCS_FILE to
    per-test temp paths so offline tests don't scribble on the real
    runtime files. Returns a restore fn that puts the originals back
    and deletes the temp files."""
    import npc_engine.story_director as sd_mod
    original_state = sd_mod.STATE_FILE
    original_ledger = sd_mod.LEDGER_FILE
    original_arcs = sd_mod.ARCS_FILE
    tmp_state = NPC_ROOT / "data" / "story_director" / f"_tmp_{tag}_state.json"
    tmp_ledger = NPC_ROOT / "data" / "story_director" / f"_tmp_{tag}_ledger.json"
    tmp_arcs = NPC_ROOT / "data" / "story_director" / f"_tmp_{tag}_arcs.json"
    for p in (tmp_state, tmp_ledger, tmp_arcs):
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass
    sd_mod.STATE_FILE = tmp_state
    sd_mod.LEDGER_FILE = tmp_ledger
    sd_mod.ARCS_FILE = tmp_arcs

    def restore():
        sd_mod.STATE_FILE = original_state
        sd_mod.LEDGER_FILE = original_ledger
        sd_mod.ARCS_FILE = original_arcs
        for p in (tmp_state, tmp_ledger, tmp_arcs):
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
    return restore


# ── Offline tests ───────────────────────────────────────────────

def test_parse_clean_json():
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    raw = '{"action": "noop", "reason": "all quiet"}'
    action = director._parse_action(raw)
    assert action == {"action": "noop", "reason": "all quiet"}, action
    print("  [PASS] parse_clean_json")


def test_parse_json_with_fences_and_prose():
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    raw = (
        "Sure, here is my decision.\n"
        "```json\n"
        '{"action": "event", "target": "all", "event": "wolves at the gate"}\n'
        "```\n"
        "Hope that works."
    )
    action = director._parse_action(raw)
    assert action["action"] == "event", action
    assert action["event"] == "wolves at the gate", action
    print("  [PASS] parse_json_with_fences_and_prose")


def test_parse_garbage_returns_noop():
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    action = director._parse_action("I have no idea what to do.")
    assert action["action"] == "noop", action
    assert "reason" in action
    print("  [PASS] parse_garbage_returns_noop")


def test_parse_coerces_mislabeled_event_to_fact():
    """0.5B Qwen regression — the model labels a fact-shaped payload as
    action=event. Coercion should rescue the intent."""
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    raw = (
        '{"action": "event", "reason": "confidant", "npc_id": "noah", '
        '"fact": "I am watching Mara.", "fact_type": "personal"}'
    )
    action = director._parse_action(raw)
    assert action["action"] == "fact", action
    assert action["npc_id"] == "noah"
    assert action["fact"] == "I am watching Mara."
    print("  [PASS] parse_coerces_mislabeled_event_to_fact")


def test_parse_coerces_mislabeled_quest_payload():
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    raw = (
        '{"action": "fact", "npc_id": "kael", '
        '"quest": {"id": "x", "name": "X", "description": "d"}}'
    )
    action = director._parse_action(raw)
    assert action["action"] == "quest", action
    print("  [PASS] parse_coerces_mislabeled_quest_payload")


def test_parse_extracts_first_of_two_json_objects():
    """Model sometimes emits two back-to-back JSON objects. Greedy regex
    would glue them into a single invalid payload; the brace scanner must
    return only the first complete object."""
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    raw = (
        '{"action": "event", "target": "all", "event": "wolves at the gate"}\n'
        '{"action": "noop", "reason": "bonus garbage"}'
    )
    action = director._parse_action(raw)
    assert action["action"] == "event", action
    assert action["event"] == "wolves at the gate", action
    print("  [PASS] parse_extracts_first_of_two_json_objects")


def test_focus_npc_picks_untouched_first():
    """When a profile has never been targeted, it wins the focus slot."""
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    # Give everyone but 'noah' a recent touch history
    director.recent_decisions = [
        {"tick": 1, "action": {"action": "event", "target": "kael", "event": "x"}},
        {"tick": 2, "action": {"action": "event", "target": "bess", "event": "y"}},
    ]
    focus = director._pick_focus_npc()
    assert focus == "noah", focus
    print("  [PASS] focus_npc_picks_untouched_first")


def test_focus_npc_rotates_by_least_recently_touched():
    """When every profile has been touched, the oldest one wins."""
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    director.recent_decisions = [
        {"tick": 1, "action": {"action": "event", "target": "bess", "event": "x"}},
        {"tick": 2, "action": {"action": "event", "target": "noah", "event": "y"}},
        {"tick": 3, "action": {"action": "event", "target": "kael", "event": "z"}},
    ]
    focus = director._pick_focus_npc()
    assert focus == "bess", focus  # oldest touch
    print("  [PASS] focus_npc_rotates_by_least_recently_touched")


def test_enforce_focus_npc_overrides_event_target():
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    action = {"action": "event", "target": "all", "event": "x"}
    out = director._enforce_focus_npc(action, "bess")
    assert out["target"] == "bess", out
    print("  [PASS] enforce_focus_npc_overrides_event_target")


def test_enforce_focus_npc_overrides_quest_npc_id():
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    action = {"action": "quest", "npc_id": "noah", "quest": {"id": "x"}}
    out = director._enforce_focus_npc(action, "bess")
    assert out["npc_id"] == "bess", out
    print("  [PASS] enforce_focus_npc_overrides_quest_npc_id")


def test_enforce_focus_npc_leaves_noop_alone():
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    action = {"action": "noop", "reason": "quiet"}
    out = director._enforce_focus_npc(action, "bess")
    assert out == {"action": "noop", "reason": "quiet"}, out
    print("  [PASS] enforce_focus_npc_leaves_noop_alone")


def test_prompt_contains_focus_block():
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    prompt = director._build_prompt("world snap", focus_npc="kael", action_kind="event")
    assert "FOCUS NPC" in prompt
    assert "kael" in prompt
    assert "MUST" in prompt
    print("  [PASS] prompt_contains_focus_block")


def test_action_kind_rotates_over_session():
    """Round-robin must advance the index each call so subsequent ticks
    don't keep picking 'event'."""
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    kinds = [director._pick_action_kind("kael") for _ in range(6)]
    # Two full laps of the rotation, should hit each kind twice
    assert kinds.count("event") == 2, kinds
    assert kinds.count("quest") == 2, kinds
    assert kinds.count("fact") == 2, kinds
    # Adjacent picks never repeat (round-robin, not randomized)
    for i in range(1, len(kinds)):
        assert kinds[i] != kinds[i - 1], kinds
    print("  [PASS] action_kind_rotates_over_session")


def test_action_kind_skips_quest_when_npc_full():
    """If the focus NPC already has >= _MAX_QUESTS_PER_NPC open quests,
    'quest' must be skipped for them — the rotation lands on the next
    valid kind instead."""
    import npc_engine.story_director as sd_mod
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    kael = engine.pie.npc_knowledge.get("kael")
    for i in range(sd_mod._MAX_QUESTS_PER_NPC):
        kael.add_quest(Quest(id=f"q{i}", name=f"q{i}", description="d"))

    # Force the rotation index to 'quest' position so we actually exercise
    # the skip path (rotation starts at 'event' by default).
    director._kind_rotation_index = _ACTION_KIND_ROTATION_INDEX_FOR("quest")
    kind = director._pick_action_kind("kael")
    assert kind != "quest", kind
    print("  [PASS] action_kind_skips_quest_when_npc_full")


def _ACTION_KIND_ROTATION_INDEX_FOR(name: str) -> int:
    import npc_engine.story_director as sd_mod
    return sd_mod._ACTION_KIND_ROTATION.index(name)


def test_enforce_action_kind_event_to_quest():
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    action = {"action": "event", "target": "kael", "event": "Something odd"}
    out = director._enforce_action_kind(action, "quest", "kael")
    assert out["action"] == "quest", out
    assert out["npc_id"] == "kael", out
    assert isinstance(out["quest"], dict), out
    assert out["quest"]["description"] == "Something odd"
    print("  [PASS] enforce_action_kind_event_to_quest")


def test_enforce_action_kind_leaves_noop_alone():
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    action = {"action": "noop", "reason": "quiet"}
    out = director._enforce_action_kind(action, "quest", "kael")
    assert out == {"action": "noop", "reason": "quiet"}, out
    print("  [PASS] enforce_action_kind_leaves_noop_alone")


def test_enforce_action_kind_event_to_fact():
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    action = {"action": "event", "target": "bess", "event": "The tax road smells of rot"}
    out = director._enforce_action_kind(action, "fact", "bess")
    assert out["action"] == "fact", out
    assert out["npc_id"] == "bess", out
    assert "rot" in out["fact"], out
    print("  [PASS] enforce_action_kind_event_to_fact")


def test_architect_plan_picks_distinct_npcs():
    """The architect must not pick the same NPC twice in one tick."""
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    plan = director._architect_plan(3)
    npcs = [p[0] for p in plan]
    assert len(npcs) == 3, plan
    assert len(set(npcs)) == 3, f"duplicate NPCs in plan: {plan}"
    print("  [PASS] architect_plan_picks_distinct_npcs")


def test_architect_plan_caps_at_npc_count():
    """If we ask for more sub-actions than there are NPCs, the plan
    truncates rather than repeating."""
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    plan = director._architect_plan(10)
    npcs = [p[0] for p in plan]
    # The stub engine has 3 NPCs; the plan can be at most 3 long.
    assert len(plan) == 3, plan
    assert len(set(npcs)) == 3, plan
    print("  [PASS] architect_plan_caps_at_npc_count")


def test_burst_rotation_holds_slot_zero_across_k_ticks():
    """Slot 0 of the architect plan must stick to the same NPC for K
    consecutive ticks, then rotate to the next least-recently-touched
    NPC on tick K+1. This is the KV cache reuse lever."""
    import npc_engine.story_director as sd_mod
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    k = sd_mod._BURST_ROTATION_DEPTH
    slot_zero: list[str] = []
    for tick_num in range(1, k + 2):  # K ticks in burst + 1 rotation tick
        plan = director._architect_plan(1)
        assert plan, f"empty plan on tick {tick_num}"
        slot_zero.append(plan[0][0])
        # Simulate what tick() does after _architect_plan returns —
        # stamp the planned NPC so subsequent rotation sees it as touched.
        director._npc_last_planned_tick[plan[0][0]] = tick_num
    # First K picks MUST all be the same NPC.
    assert len(set(slot_zero[:k])) == 1, f"burst broke early: {slot_zero[:k]}"
    # Tick K+1 must have rotated to a different NPC.
    assert slot_zero[k] != slot_zero[0], (
        f"burst failed to rotate after {k} ticks: {slot_zero}"
    )
    print("  [PASS] burst_rotation_holds_slot_zero_across_k_ticks")


def test_burst_rotation_coexists_with_multi_action_rotation():
    """In multi-action mode the non-zero slots must still rotate
    normally while slot 0 is held by burst. Distinct NPCs per tick
    across all slots is still required (no worker collisions)."""
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    # Two ticks of 3 actions each. Slot 0 should repeat across the
    # two ticks; slots 1 and 2 should change to give fresh coverage.
    plan_a = director._architect_plan(3)
    assert len(plan_a) == 3, plan_a
    assert len({p[0] for p in plan_a}) == 3, f"duplicate NPCs in tick 1: {plan_a}"
    # Simulate tick()'s planned-tick bump for every slot so rotation
    # on the next call sees them as touched.
    for npc, _ in plan_a:
        director._npc_last_planned_tick[npc] = 1

    plan_b = director._architect_plan(3)
    assert len(plan_b) == 3, plan_b
    assert len({p[0] for p in plan_b}) == 3, f"duplicate NPCs in tick 2: {plan_b}"
    # Slot 0 must be the SAME NPC across ticks (burst held it).
    assert plan_b[0][0] == plan_a[0][0], (
        f"slot 0 must stick: tick1={plan_a[0][0]} tick2={plan_b[0][0]}"
    )
    print("  [PASS] burst_rotation_coexists_with_multi_action_rotation")


def test_burst_rotation_persists_across_director_instances():
    """Burst focus + remaining must survive a save/load cycle so a
    bench resume (or crash recovery) continues the same cache-reuse
    window instead of restarting the burst from scratch."""
    restore = _isolate_state_file("burst_persist")
    try:
        engine1 = _make_stub_engine()
        director1 = StoryDirector(engine1)
        # Tick once so burst gets locked in + state gets saved.
        director1._architect_plan(1)
        expected_focus = director1._burst_focus_npc
        expected_remaining = director1._burst_remaining
        assert expected_focus is not None
        assert expected_remaining > 0
        director1._save_state()

        # Fresh director reads state.json back.
        engine2 = _make_stub_engine()
        director2 = StoryDirector(engine2)
        assert director2._burst_focus_npc == expected_focus, (
            director2._burst_focus_npc, expected_focus,
        )
        assert director2._burst_remaining == expected_remaining, (
            director2._burst_remaining, expected_remaining,
        )
    finally:
        restore()
    print("  [PASS] burst_rotation_persists_across_director_instances")


def test_burst_rotation_recovers_when_focus_npc_vanishes():
    """If the persisted burst focus no longer exists in the live cast
    (e.g. an NPC was removed between runs), the director must drop
    the stale burst and pick a fresh NPC via normal rotation instead
    of returning None or crashing."""
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    # Simulate a stale burst pointing at an NPC that was deleted.
    director._burst_focus_npc = "ghost_npc_id"
    director._burst_remaining = 3
    plan = director._architect_plan(1)
    assert plan, "plan should not be empty when live NPCs exist"
    assert plan[0][0] in {"kael", "bess", "noah"}, plan
    # The stale id should be replaced by a real one.
    assert director._burst_focus_npc != "ghost_npc_id", (
        director._burst_focus_npc,
    )
    print("  [PASS] burst_rotation_recovers_when_focus_npc_vanishes")


def test_pick_focus_respects_extra_exclude():
    """The in-flight architect planner uses extra_exclude to mark NPCs
    as taken. _pick_focus_npc must honor it across both layers (player
    reactivity + rotation)."""
    restore = _isolate_state_file("exclude")
    try:
        engine = _make_stub_engine()
        director = StoryDirector(engine)
        # Even with a pending player action targeting bess, exclude={bess}
        # should fall through to rotation
        director.recent_player_actions = [{
            "at": "2026-04-13T10:05:00+00:00",
            "tick_at_time": 1,
            "text": "Player gave bess a gift",
            "target": "bess",
        }]
        focus = director._pick_focus_npc(extra_exclude={"bess"})
        assert focus != "bess", focus
        assert focus in {"kael", "noah"}, focus
    finally:
        restore()
    print("  [PASS] pick_focus_respects_extra_exclude")


def test_multi_action_tick_runs_n_workers():
    """An actions_per_tick=3 tick should produce 3 distinct sub-actions
    and update tick_count by exactly one (not three — it's still ONE
    tick, just with parallel workers). Uses a 7-NPC cast so the
    small-cast cap doesn't fire."""
    restore = _isolate_state_file("multi")
    try:
        engine = _make_stub_engine(
            profile_specs=_LARGE_STUB_PROFILES,
            responses=[
                '{"action": "event", "target": "kael", "event": "kael event"}',
                '{"action": "event", "target": "bess", "event": "bess event"}',
                '{"action": "event", "target": "noah", "event": "noah event"}',
            ],
        )
        director = StoryDirector(engine)
        result = director.tick(actions_per_tick=3)
        assert "sub_actions" in result, result
        assert len(result["sub_actions"]) == 3, result
        assert director.tick_count == 1, director.tick_count
        targets = [r["focus_npc"] for r in result["sub_actions"]]
        assert len(set(targets)) == 3, targets
    finally:
        restore()
    print("  [PASS] multi_action_tick_runs_n_workers")


def test_small_cast_caps_actions_per_tick_to_one():
    """On a small cast (<=_SMALL_CAST_THRESHOLD NPCs), tick() forces
    actions_per_tick down to 1 regardless of the caller's request.
    Prevents the PB-style bloat where 3 actions/tick × 10 ticks piles
    10 injections onto each NPC."""
    import npc_engine.story_director as sd_mod
    restore = _isolate_state_file("smallcast_cap")
    try:
        # Default 3-NPC cast is below threshold.
        assert len(_DEFAULT_STUB_PROFILES) <= sd_mod._SMALL_CAST_THRESHOLD
        engine = _make_stub_engine(responses=[
            '{"action": "event", "target": "kael", "event": "only one"}',
        ])
        director = StoryDirector(engine)
        # Caller asks for 3, cap should trim to 1.
        result = director.tick(actions_per_tick=3)
        # Single-action mode returns the legacy flat shape.
        assert "action" in result, result
        assert "sub_actions" not in result, result
        assert director.tick_count == 1, director.tick_count
    finally:
        restore()
    print("  [PASS] small_cast_caps_actions_per_tick_to_one")


def test_large_cast_honors_requested_actions_per_tick():
    """On a cast above _SMALL_CAST_THRESHOLD, the caller's requested
    actions_per_tick passes through unchanged."""
    import npc_engine.story_director as sd_mod
    restore = _isolate_state_file("largecast_pass")
    try:
        assert len(_LARGE_STUB_PROFILES) > sd_mod._SMALL_CAST_THRESHOLD
        engine = _make_stub_engine(
            profile_specs=_LARGE_STUB_PROFILES,
            responses=[
                '{"action": "event", "target": "kael", "event": "a"}',
                '{"action": "event", "target": "bess", "event": "b"}',
                '{"action": "event", "target": "noah", "event": "c"}',
            ],
        )
        director = StoryDirector(engine)
        result = director.tick(actions_per_tick=3)
        assert "sub_actions" in result, result
        assert len(result["sub_actions"]) == 3, result
    finally:
        restore()
    print("  [PASS] large_cast_honors_requested_actions_per_tick")


def test_small_cast_cap_threshold_boundary():
    """A cast of exactly _SMALL_CAST_THRESHOLD NPCs must still trigger
    the cap (threshold is inclusive). One NPC above the threshold must
    NOT trigger it."""
    import npc_engine.story_director as sd_mod
    # Build a cast of exactly _SMALL_CAST_THRESHOLD.
    at_threshold = tuple(
        (f"npc{i}", "role") for i in range(sd_mod._SMALL_CAST_THRESHOLD)
    )
    above_threshold = at_threshold + (("extra_npc", "role"),)

    restore = _isolate_state_file("cap_at_threshold")
    try:
        engine = _make_stub_engine(
            profile_specs=at_threshold,
            responses=[
                '{"action": "event", "target": "npc0", "event": "x"}',
            ],
        )
        director = StoryDirector(engine)
        result = director.tick(actions_per_tick=3)
        assert "sub_actions" not in result, (
            f"cast={sd_mod._SMALL_CAST_THRESHOLD} should have been capped: {result}"
        )
    finally:
        restore()

    restore = _isolate_state_file("cap_above_threshold")
    try:
        engine = _make_stub_engine(
            profile_specs=above_threshold,
            responses=[
                '{"action": "event", "target": "npc0", "event": "x"}',
                '{"action": "event", "target": "npc1", "event": "y"}',
                '{"action": "event", "target": "npc2", "event": "z"}',
            ],
        )
        director = StoryDirector(engine)
        result = director.tick(actions_per_tick=3)
        assert "sub_actions" in result, result
        assert len(result["sub_actions"]) == 3, result
    finally:
        restore()
    print("  [PASS] small_cast_cap_threshold_boundary")


def test_small_cast_single_action_passthrough():
    """When caller already requests 1 action/tick on a small cast, the
    cap is a no-op — no log noise, normal single-action flow."""
    restore = _isolate_state_file("smallcast_passthrough")
    try:
        engine = _make_stub_engine(responses=[
            '{"action": "event", "target": "kael", "event": "x"}',
        ])
        director = StoryDirector(engine)
        result = director.tick(actions_per_tick=1)
        assert "action" in result, result
        assert "sub_actions" not in result, result
    finally:
        restore()
    print("  [PASS] small_cast_single_action_passthrough")


# ── Zone layer tests (Phase 1) ───────────────────────────────────

_ZONED_PROFILES = (
    # 4 dock_district NPCs + 2 lighthouse_bluffs NPCs + 1 mobile fence
    # (starts in dock_district). Matches the shape of
    # data/worlds/port_blackwater_zoned.
    ("reva",      "harbor_master", "dock_district",    False),
    ("finn",      "dock_worker",   "dock_district",    False),
    ("old_bones", "tavern_owner",  "dock_district",    False),
    ("varro",     "fence",         "dock_district",    True),
    ("thessa",    "hermit",        "lighthouse_bluffs", False),
    ("brom",      "pilgrim",       "lighthouse_bluffs", False),
)


def test_empty_active_zones_preserves_world_wide_mode():
    """Backward-compat critical path: when _active_zones is empty,
    _pick_focus_npc returns exactly what the pre-zone code would have
    picked. Every pre-zone bench and test still passes because of
    this invariant."""
    restore = _isolate_state_file("empty_zones")
    try:
        engine = _make_stub_engine(profile_specs=_ZONED_PROFILES)
        director = StoryDirector(engine)
        assert director._active_zones == set()
        # With no zone restriction, the rotation pool is all 6 NPCs.
        # First pick with empty _npc_last_planned_tick and no
        # exclude should return the first NPC in insertion order
        # (reva), identical to non-zone rotation.
        focus = director._pick_focus_npc()
        assert focus == "reva", focus
    finally:
        restore()
    print("  [PASS] empty_active_zones_preserves_world_wide_mode")


def test_zone_soft_weighting_prefers_in_zone():
    """With active_zones = {dock_district}, rotation should strongly
    prefer dock NPCs (in-zone) over lighthouse NPCs (out-of-zone).
    Over many picks, in-zone should dominate."""
    restore = _isolate_state_file("zone_soft")
    try:
        import npc_engine.story_director as sd_mod
        engine = _make_stub_engine(profile_specs=_ZONED_PROFILES)
        director = StoryDirector(engine)
        director.set_active_zones(["dock_district"])
        in_zone_ids = {"reva", "finn", "old_bones", "varro"}
        out_of_zone_ids = {"thessa", "brom"}
        in_count = 0
        out_count = 0
        # Sample many picks to exercise the 85/15 statistical split.
        # After each pick, bump _npc_last_planned_tick so rotation
        # advances rather than oscillating on the same NPC.
        for tick_num in range(1, 101):
            focus = director._pick_focus_npc()
            assert focus is not None
            if focus in in_zone_ids:
                in_count += 1
            elif focus in out_of_zone_ids:
                out_count += 1
            else:
                raise AssertionError(f"unexpected focus: {focus}")
            director._npc_last_planned_tick[focus] = tick_num
        # At _OUT_OF_ZONE_RATE = 7, we expect ~100*6/7 in-zone and
        # ~100/7 out-of-zone. Allow wide tolerance for rotation
        # order variance.
        assert in_count >= 80, f"in={in_count} out={out_count}"
        assert out_count >= 8, f"in={in_count} out={out_count}"
        assert out_count <= 20, f"in={in_count} out={out_count}"
    finally:
        restore()
    print("  [PASS] zone_soft_weighting_prefers_in_zone")


def test_zone_out_of_zone_rate_honored_precisely():
    """The out-of-zone escape hatch fires on counter % _OUT_OF_ZONE_RATE
    == 0. Every Nth _pick_focus_npc call should hit an out-of-zone NPC
    when one is available (unless the in-zone pool is exhausted)."""
    restore = _isolate_state_file("zone_rate")
    try:
        import npc_engine.story_director as sd_mod
        engine = _make_stub_engine(profile_specs=_ZONED_PROFILES)
        director = StoryDirector(engine)
        director.set_active_zones(["dock_district"])
        # First N-1 picks should be in-zone; Nth pick (counter=N)
        # should be out of zone. Cross-check against the constant
        # so this test stays in sync if the rate changes.
        rate = sd_mod._OUT_OF_ZONE_RATE
        in_zone_ids = {"reva", "finn", "old_bones", "varro"}
        out_of_zone_ids = {"thessa", "brom"}
        for i in range(1, rate + 1):
            focus = director._pick_focus_npc()
            director._npc_last_planned_tick[focus] = i
            if i < rate:
                assert focus in in_zone_ids, (
                    f"pick {i} should be in-zone, got {focus}"
                )
            else:
                # The rate-th pick should jump out of zone.
                assert focus in out_of_zone_ids, (
                    f"pick {i} should be out-of-zone, got {focus}"
                )
    finally:
        restore()
    print("  [PASS] zone_out_of_zone_rate_honored_precisely")


def test_global_zone_npc_always_in_scope():
    """NPCs with zone='global' (or no zone field) should always pass
    the _npc_in_active_zone check, regardless of what the active
    zones are."""
    restore = _isolate_state_file("zone_global")
    try:
        mixed = (
            ("local_reva",  "harbor_master", "dock_district"),
            ("global_god",  "deity",         "global"),
            ("far_thessa",  "hermit",        "lighthouse_bluffs"),
        )
        engine = _make_stub_engine(profile_specs=mixed)
        director = StoryDirector(engine)
        director.set_active_zones(["dock_district"])
        assert director._npc_in_active_zone("local_reva") is True
        assert director._npc_in_active_zone("global_god") is True
        assert director._npc_in_active_zone("far_thessa") is False
    finally:
        restore()
    print("  [PASS] global_zone_npc_always_in_scope")


def test_quest_hard_filter_on_out_of_zone_focus():
    """If the focus NPC is out of the active zone set, 'quest' gets
    dropped from the allowed action kinds — rotation falls through to
    event or fact. Out-of-zone NPCs never offer quests to the player."""
    restore = _isolate_state_file("zone_quest_filter")
    try:
        engine = _make_stub_engine(profile_specs=_ZONED_PROFILES)
        director = StoryDirector(engine)
        director.set_active_zones(["dock_district"])
        # Force rotation index to 'quest' slot so the natural pick
        # would be quest. Out-of-zone focus should block it.
        import npc_engine.story_director as sd_mod
        director._kind_rotation_index = sd_mod._ACTION_KIND_ROTATION.index("quest")
        # thessa is in lighthouse_bluffs — out of zone
        kind = director._pick_action_kind("thessa")
        assert kind != "quest", (
            f"out-of-zone focus should not receive quest, got {kind}"
        )
        # reva is in dock_district — in zone, should still receive quest
        director._kind_rotation_index = sd_mod._ACTION_KIND_ROTATION.index("quest")
        kind = director._pick_action_kind("reva")
        assert kind == "quest", kind
    finally:
        restore()
    print("  [PASS] quest_hard_filter_on_out_of_zone_focus")


def test_bounded_snapshot_surfaces_active_zone_npcs():
    """When the cast exceeds _SNAPSHOT_BOUND_THRESHOLD, the bounded
    snapshot selector must include active-zone NPCs in its priority
    tiers so the LLM sees them even when the full cast doesn't fit."""
    restore = _isolate_state_file("zone_snapshot")
    try:
        import npc_engine.story_director as sd_mod
        # Build a 40-NPC cast where half are in dock_district and the
        # rest are in "far_away". We want the bounded selector to pick
        # dock NPCs over far ones when active_zones is set.
        specs = []
        for i in range(20):
            specs.append((f"dock_{i}", "worker", "dock_district"))
        for i in range(20):
            specs.append((f"far_{i}", "wanderer", "far_away"))
        engine = _make_stub_engine(profile_specs=tuple(specs))
        director = StoryDirector(engine)
        director.set_active_zones(["dock_district"])
        selected = director._select_snapshot_npcs(
            list(engine.pie.npc_knowledge.profiles.keys()),
            planned_focus_ids=[],
        )
        dock_in_selection = sum(1 for n in selected if n.startswith("dock_"))
        far_in_selection = sum(1 for n in selected if n.startswith("far_"))
        cap = StoryDirector._SNAPSHOT_NPC_CAP
        # The cap is 16 — we expect all 16 to be dock NPCs, since the
        # dock zone has 20 candidates and fills the cap.
        assert dock_in_selection == cap, (
            f"expected all {cap} slots to be dock NPCs, "
            f"got dock={dock_in_selection} far={far_in_selection}"
        )
        assert far_in_selection == 0, far_in_selection
    finally:
        restore()
    print("  [PASS] bounded_snapshot_surfaces_active_zone_npcs")


def test_npc_current_zone_mutation_persists():
    """set_npc_current_zone on a mobile NPC should update current_zone,
    persist to state.json, and survive a reload into a fresh director."""
    restore = _isolate_state_file("zone_persist")
    try:
        engine = _make_stub_engine(profile_specs=_ZONED_PROFILES)
        director = StoryDirector(engine)
        # Varro is mobile, starts in dock_district per the stub spec.
        result = director.set_npc_current_zone("varro", "lighthouse_bluffs")
        assert result["ok"] is True, result
        assert engine.pie.npc_knowledge.profiles["varro"].current_zone == "lighthouse_bluffs"
        # set_active_zones is also persistable — test round-trip.
        director.set_active_zones(["dock_district", "lighthouse_bluffs"])
        director._save_state()

        # Fresh director reads the saved state. The NPC's current_zone
        # lives on the NPC object, not state.json, so a fresh engine
        # would re-load varro at "dock_district" (YAML default).
        # Active zones DO persist via state.json.
        engine2 = _make_stub_engine(profile_specs=_ZONED_PROFILES)
        director2 = StoryDirector(engine2)
        assert director2._active_zones == {"dock_district", "lighthouse_bluffs"}, (
            director2._active_zones
        )
    finally:
        restore()
    print("  [PASS] npc_current_zone_mutation_persists")


def test_immobile_npc_zone_change_rejected():
    """Stationary NPCs (mobile=False) must refuse runtime zone
    changes. Game client gets a clear error so it knows to mark the
    profile mobile in YAML first."""
    restore = _isolate_state_file("zone_immobile")
    try:
        engine = _make_stub_engine(profile_specs=_ZONED_PROFILES)
        director = StoryDirector(engine)
        # reva is stationary
        result = director.set_npc_current_zone("reva", "lighthouse_bluffs")
        assert result["ok"] is False
        assert "not mobile" in result.get("reason", "")
        # Zone didn't change
        assert engine.pie.npc_knowledge.profiles["reva"].current_zone == "dock_district"
    finally:
        restore()
    print("  [PASS] immobile_npc_zone_change_rejected")


def test_zone_change_breaks_burst_rotation():
    """When the active zone set changes and the current burst focus
    is an NPC no longer in any active zone, the burst breaks so the
    next tick picks a fresh in-zone NPC."""
    restore = _isolate_state_file("zone_burst_break")
    try:
        engine = _make_stub_engine(profile_specs=_ZONED_PROFILES)
        director = StoryDirector(engine)
        director.set_active_zones(["dock_district"])
        # Seed a burst on finn (in dock_district)
        director._burst_focus_npc = "finn"
        director._burst_remaining = 3
        # Player moves to lighthouse
        director.set_active_zones(["lighthouse_bluffs"])
        # Burst should have broken since finn is no longer in an
        # active zone.
        assert director._burst_focus_npc is None, director._burst_focus_npc
        assert director._burst_remaining == 0
    finally:
        restore()
    print("  [PASS] zone_change_breaks_burst_rotation")


def test_burst_rotation_locks_to_in_zone():
    """When active_zones is set, burst rotation must never pick an
    out-of-zone NPC as its sticky focus. The out-of-zone escape hatch
    should only fire on non-burst slots (slots 1+ of the architect
    plan) — otherwise a single escape pick gets amplified into
    K_BURST_ROTATION_DEPTH ticks of distant focus."""
    import npc_engine.story_director as sd_mod
    restore = _isolate_state_file("burst_zone_lock")
    try:
        engine = _make_stub_engine(profile_specs=_ZONED_PROFILES)
        director = StoryDirector(engine)
        director.set_active_zones(["dock_district"])
        in_zone_ids = {"reva", "finn", "old_bones", "varro"}
        # Crank the escape counter so the NEXT pick without zone_lock_in
        # WOULD fire the out-of-zone escape. Burst rotation must still
        # land in-zone.
        director._zone_escape_counter = sd_mod._OUT_OF_ZONE_RATE - 1
        # Drive several ticks through _architect_plan and verify slot 0
        # stays in-zone across burst transitions.
        for tick_num in range(1, 30):
            plan = director._architect_plan(1)
            if not plan:
                break
            focus = plan[0][0]
            assert focus in in_zone_ids, (
                f"burst slot 0 picked out-of-zone NPC {focus} on tick {tick_num}"
            )
            director._npc_last_planned_tick[focus] = tick_num
    finally:
        restore()
    print("  [PASS] burst_rotation_locks_to_in_zone")


# ── Lifecycle: death plumbing tests (Phase 2a) ─────────────────

def _seed_zoned_director(tag: str):
    """Shared helper: fresh zoned director with isolated state."""
    restore = _isolate_state_file(tag)
    engine = _make_stub_engine(profile_specs=_ZONED_PROFILES)
    director = StoryDirector(engine)
    return engine, director, restore


def test_queue_death_request_validates_alive_and_exists():
    """queue_death_request rejects unknown NPCs and already-dead
    NPCs with clear error messages."""
    engine, director, restore = _seed_zoned_director("death_queue_valid")
    try:
        result = director.queue_death_request("nobody_id")
        assert result["ok"] is False
        assert "unknown" in result["reason"]
        # Kill reva manually then try to kill her again
        engine.pie.npc_knowledge.profiles["reva"].status = "deceased"
        result = director.queue_death_request("reva")
        assert result["ok"] is False
        assert "not alive" in result["reason"]
    finally:
        restore()
    print("  [PASS] queue_death_request_validates_alive_and_exists")


def test_death_dispatch_marks_npc_deceased_and_breaks_burst():
    """A death dispatch must flip status to deceased, stamp
    death_tick/death_cause, drop the burst focus if it matches, and
    record the death in _deceased_npcs."""
    engine, director, restore = _seed_zoned_director("death_dispatch")
    try:
        director._burst_focus_npc = "reva"
        director._burst_remaining = 3
        result = director._dispatch_npc_death(
            npc_id="reva", cause="bandit ambush",
            transfers_quests_to=None,
        )
        assert result["ok"] is True
        npc = engine.pie.npc_knowledge.profiles["reva"]
        assert npc.status == "deceased"
        assert npc.death_cause == "bandit ambush"
        assert npc.death_tick == 1
        # Burst broke
        assert director._burst_focus_npc is None
        assert director._burst_remaining == 0
        # Deceased record exists
        assert "reva" in director._deceased_npcs
        rec = director._deceased_npcs["reva"]
        assert rec["death_cause"] == "bandit ambush"
        assert rec["name"] == "Reva"
    finally:
        restore()
    print("  [PASS] death_dispatch_marks_npc_deceased_and_breaks_burst")


def test_deceased_npc_excluded_from_focus_selection():
    """_pick_focus_npc must never return a deceased NPC, even if
    they're in the alphabetical-first slot in the rotation."""
    engine, director, restore = _seed_zoned_director("death_focus_exclude")
    try:
        # Kill reva (rotation's alphabetically-first dock NPC)
        engine.pie.npc_knowledge.profiles["reva"].status = "deceased"
        for _ in range(20):
            focus = director._pick_focus_npc()
            assert focus is not None
            assert focus != "reva"
            director._npc_last_planned_tick[focus] = director._npc_last_planned_tick.get(focus, 0) + 1
    finally:
        restore()
    print("  [PASS] deceased_npc_excluded_from_focus_selection")


def test_death_transfers_open_quests_to_inheritor():
    """When transfers_quests_to is set, open quests from the deceased
    NPC move to the inheritor with inherited=True tagged in the
    dispatch record."""
    engine, director, restore = _seed_zoned_director("death_inherit")
    try:
        from npc_engine.knowledge import Quest
        # Give reva an open quest
        engine.pie.npc_knowledge.profiles["reva"].add_quest(
            Quest(id="keep_port_safe", name="Keep the Port Safe",
                   description="Protect the harbor from pirates")
        )
        pre_finn_quests = len(engine.pie.npc_knowledge.profiles["finn"].quests)
        result = director._dispatch_npc_death(
            npc_id="reva", cause="drowned",
            transfers_quests_to="finn",
        )
        assert result["ok"] is True
        # Finn inherited the quest
        post_finn_quests = len(engine.pie.npc_knowledge.profiles["finn"].quests)
        assert post_finn_quests == pre_finn_quests + 1
        quest_records = result["record"]["quests_cleaned"]
        assert any(
            q["quest_id"] == "keep_port_safe" and q["transition"] == "transferred"
            for q in quest_records
        )
    finally:
        restore()
    print("  [PASS] death_transfers_open_quests_to_inheritor")


def test_death_aborts_open_quests_without_inheritor():
    """When there's no inheritor, open quests get marked aborted
    with giver_deceased as the reason."""
    engine, director, restore = _seed_zoned_director("death_abort")
    try:
        from npc_engine.knowledge import Quest
        reva = engine.pie.npc_knowledge.profiles["reva"]
        reva.add_quest(Quest(id="orphan_quest", name="Orphan",
                              description="No inheritor for this one"))
        result = director._dispatch_npc_death(
            npc_id="reva", cause="fell overboard",
            transfers_quests_to=None,
        )
        assert result["ok"] is True
        # Quest still on the deceased NPC but marked aborted
        assert any(
            q.id == "orphan_quest" and q.status == "aborted"
            for q in reva.quests
        )
        quest_records = result["record"]["quests_cleaned"]
        assert any(
            q["quest_id"] == "orphan_quest" and q["transition"] == "aborted"
            for q in quest_records
        )
    finally:
        restore()
    print("  [PASS] death_aborts_open_quests_without_inheritor")


def test_arc_on_cast_death_resolves_post_confront_arc():
    """An arc already past the confront beat (current_beat >= 2)
    should resolve immediately when a cast member dies — the death
    IS the resolution."""
    engine, director, restore = _seed_zoned_director("death_arc_resolve")
    try:
        from npc_engine.story_director import NarrativeArc
        arc = NarrativeArc(
            id="test_arc", theme="Reva's fate",
            focus_npcs=["reva"], beat_goals=["seed", "escalate", "confront", "resolve"],
            current_beat=2,  # at confront
            status="active", started_at_tick=1,
        )
        director.arc_planner.arcs.append(arc)
        director.arc_planner.active_arc_ids.append("test_arc")
        result = director.arc_planner.on_cast_death(
            npc_id="reva", current_tick=5,
        )
        assert arc.status == "resolved"
        assert "test_arc" not in director.arc_planner.active_arc_ids
        assert len(result["arcs_affected"]) == 1
        assert result["arcs_affected"][0]["transition"] == "resolved_by_death"
    finally:
        restore()
    print("  [PASS] arc_on_cast_death_resolves_post_confront_arc")


def test_arc_on_cast_death_trims_cast_on_early_arc():
    """An arc still at seed or escalate with multiple cast members
    should drop the deceased NPC from its cast and keep running."""
    engine, director, restore = _seed_zoned_director("death_arc_trim")
    try:
        from npc_engine.story_director import NarrativeArc
        arc = NarrativeArc(
            id="trim_arc", theme="Dock trio",
            focus_npcs=["reva", "finn", "old_bones"],
            beat_goals=["seed", "escalate", "confront", "resolve"],
            current_beat=1,  # at escalate, still early
            status="active", started_at_tick=1,
            touches_since_last_advance=3,
        )
        director.arc_planner.arcs.append(arc)
        director.arc_planner.active_arc_ids.append("trim_arc")
        result = director.arc_planner.on_cast_death(
            npc_id="reva", current_tick=4,
        )
        assert arc.status == "active"
        assert "reva" not in arc.focus_npcs
        assert set(arc.focus_npcs) == {"finn", "old_bones"}
        # Touch counter resets so we earn the next beat with the
        # smaller cast
        assert arc.touches_since_last_advance == 0
        assert result["arcs_affected"][0]["transition"] == "cast_trimmed"
    finally:
        restore()
    print("  [PASS] arc_on_cast_death_trims_cast_on_early_arc")


def test_lifecycle_tick_drains_pending_death():
    """A queued death request must be dispatched on the next
    lifecycle tick. The NPC flips to deceased before the architect
    plans its focus picks."""
    restore = _isolate_state_file("lifecycle_drain")
    try:
        engine = _make_stub_engine(
            profile_specs=_ZONED_PROFILES,
            responses=[
                '{"action": "event", "target": "reva", "event": "x"}'
                for _ in range(5)
            ],
        )
        director = StoryDirector(engine)
        director.queue_death_request("reva", cause="stabbed",
                                      transfers_quests_to=None)
        assert len(director._pending_death_requests) == 1
        # Run a tick — lifecycle should drain and dispatch the death
        director.tick()
        assert len(director._pending_death_requests) == 0
        assert engine.pie.npc_knowledge.profiles["reva"].status == "deceased"
        assert "reva" in director._deceased_npcs
    finally:
        restore()
    print("  [PASS] lifecycle_tick_drains_pending_death")


def test_recently_departed_section_in_snapshot():
    """When a deceased NPC is in an active arc's cast OR died within
    the last 10 ticks, the world snapshot should include a
    RECENTLY DEPARTED section."""
    engine, director, restore = _seed_zoned_director("death_snapshot")
    try:
        director.tick_count = 5
        director._dispatch_npc_death(
            npc_id="finn", cause="accident at the docks",
            transfers_quests_to=None,
        )
        snap = director._world_snapshot()
        assert "RECENTLY DEPARTED" in snap, snap
        assert "finn" in snap.lower() or "Finn" in snap
        assert "accident at the docks" in snap
        # The live roster should NOT include finn
        assert "- finn (dock_worker)" not in snap
    finally:
        restore()
    print("  [PASS] recently_departed_section_in_snapshot")


def test_postgen_detect_deceased_reference_catches_living_action():
    """detect_deceased_reference should return the dead NPC's name
    when the dialogue treats them as an active subject, and None
    when the dialogue already frames the death."""
    from npc_engine.postgen import detect_deceased_reference
    deceased = ["Kael", "Mara"]
    # Living-action: should catch
    d1 = "Kael walks into the tavern and greets everyone."
    assert detect_deceased_reference(d1, deceased) == "Kael"
    d2 = "Hello Mara, I have news from the road."
    assert detect_deceased_reference(d2, deceased) == "Mara"
    # Acknowledged: should NOT catch
    d3 = "The late Kael used to drink at this very table."
    assert detect_deceased_reference(d3, deceased) is None
    d4 = "Kael was a fine blacksmith before the bandits came."
    assert detect_deceased_reference(d4, deceased) is None
    d5 = "We visit Mara's grave on the solstice."
    assert detect_deceased_reference(d5, deceased) is None
    # No dead NPC mentioned
    d6 = "The harbor is busy today."
    assert detect_deceased_reference(d6, deceased) is None
    print("  [PASS] postgen_detect_deceased_reference_catches_living_action")


def test_postgen_repair_deceased_reference_adds_late_prefix():
    """repair_deceased_reference wraps the first name occurrence in
    'the late {name}' and optionally appends the death cause."""
    from npc_engine.postgen import repair_deceased_reference
    out1 = repair_deceased_reference(
        "Kael walks into the tavern.",
        deceased_name="Kael", death_cause="",
    )
    assert "the late Kael" in out1
    assert "walks into the tavern" in out1

    out2 = repair_deceased_reference(
        "Kael greets the patrons, Kael smiles.",
        deceased_name="Kael", death_cause="plague",
    )
    # First occurrence wrapped, second left alone
    assert "the late Kael (plague)" in out2
    assert out2.count("the late Kael") == 1
    print("  [PASS] postgen_repair_deceased_reference_adds_late_prefix")


# ── Lifecycle: birth pipeline tests (Phase 2b) ─────────────────

def test_birth_dispatch_creates_profile_and_registers():
    """_dispatch_npc_birth must write a YAML file to the world's
    npc_profiles dir and call engine.add_profile to register the
    new NPC at runtime. The new NPC should appear in
    engine.pie.npc_knowledge.profiles afterward."""
    import tempfile, shutil
    restore = _isolate_state_file("birth_dispatch")
    # Create a temp world dir so profile files go somewhere safe.
    tmp_world = Path(tempfile.mkdtemp(prefix="test_birth_"))
    (tmp_world / "npc_profiles").mkdir()
    try:
        engine = _make_stub_engine(profile_specs=_ZONED_PROFILES)
        # Point the engine's world_dir at the temp dir so the birth
        # dispatch writes profiles there (not into the real data/).
        engine.config = SimpleNamespace(
            world_name="TestWorld", world_dir=str(tmp_world),
        )
        director = StoryDirector(engine)
        director._zone_config = {
            "dock_district": {
                "target_population": 5,
                "min_population": 3,
                "role_pool": ["merchant", "guard"],
                "lore_hook": "test dock",
            },
        }
        request = {"zone": "dock_district", "role": "merchant", "reason": "test"}
        result = director._dispatch_npc_birth(request)
        assert result["ok"] is True, result
        npc_id = result["record"]["npc_id"]
        assert npc_id in engine.pie.npc_knowledge.profiles, (
            f"new NPC {npc_id} not found in profiles"
        )
        # Profile file exists on disk
        profile_path = Path(result["record"]["profile_path"])
        assert profile_path.exists(), f"profile file not written: {profile_path}"
        # Birth history recorded
        assert len(director._birth_history) == 1
        assert director._birth_history[0]["npc_id"] == npc_id
    finally:
        restore()
        shutil.rmtree(tmp_world, ignore_errors=True)
    print("  [PASS] birth_dispatch_creates_profile_and_registers")


def test_population_gap_triggers_birth_on_lifecycle_tick():
    """When a zone is below min_population, the lifecycle tick
    should auto-generate a birth to fill the gap without the game
    client requesting it explicitly."""
    import tempfile, shutil
    restore = _isolate_state_file("birth_pop_gap")
    tmp_world = Path(tempfile.mkdtemp(prefix="test_pop_"))
    (tmp_world / "npc_profiles").mkdir()
    try:
        # Start with 2 dock NPCs — below the min of 3
        two_dock = (
            ("finn", "dock_worker", "dock_district"),
            ("old_bones", "tavern_owner", "dock_district"),
        )
        engine = _make_stub_engine(
            profile_specs=two_dock,
            responses=[
                '{"action": "event", "target": "finn", "event": "x"}',
            ],
        )
        engine.config = SimpleNamespace(
            world_name="TestWorld", world_dir=str(tmp_world),
        )
        director = StoryDirector(engine)
        director._zone_config = {
            "dock_district": {
                "target_population": 4,
                "min_population": 3,
                "role_pool": ["merchant", "guard"],
                "lore_hook": "test dock",
            },
        }
        pre_count = len(engine.pie.npc_knowledge.profiles)
        # tick() calls _lifecycle_tick which should detect the gap
        # and fire a birth before the architect plan runs.
        director.tick()
        post_count = len(engine.pie.npc_knowledge.profiles)
        assert post_count == pre_count + 1, (
            f"expected 1 birth, got {post_count - pre_count}"
        )
    finally:
        restore()
        shutil.rmtree(tmp_world, ignore_errors=True)
    print("  [PASS] population_gap_triggers_birth_on_lifecycle_tick")


def test_population_at_target_no_birth():
    """When every zone is at or above min_population, the lifecycle
    tick must NOT auto-generate a birth."""
    restore = _isolate_state_file("birth_no_gap")
    try:
        engine = _make_stub_engine(profile_specs=_ZONED_PROFILES)
        director = StoryDirector(engine)
        # 4 dock NPCs (reva, finn, old_bones, varro) vs min=3 → no gap
        director._zone_config = {
            "dock_district": {
                "target_population": 5,
                "min_population": 3,
                "role_pool": ["merchant"],
                "lore_hook": "test",
            },
        }
        gap = director._find_population_gap()
        assert gap is None, f"expected no gap, got {gap}"
    finally:
        restore()
    print("  [PASS] population_at_target_no_birth")


def test_queue_birth_request_queues_and_drains():
    """An explicit birth request via queue_birth_request must queue
    immediately and drain on the next lifecycle tick, taking
    priority over auto-gap fills."""
    import tempfile, shutil
    restore = _isolate_state_file("birth_queue")
    tmp_world = Path(tempfile.mkdtemp(prefix="test_bq_"))
    (tmp_world / "npc_profiles").mkdir()
    try:
        engine = _make_stub_engine(
            profile_specs=_ZONED_PROFILES,
            responses=[
                '{"action": "event", "target": "finn", "event": "x"}',
            ],
        )
        engine.config = SimpleNamespace(
            world_name="TestWorld", world_dir=str(tmp_world),
        )
        director = StoryDirector(engine)
        result = director.queue_birth_request(
            zone="lighthouse_bluffs", role="keeper",
        )
        assert result["ok"] is True
        assert len(director._pending_birth_requests) == 1
        # tick() drains it
        director.tick()
        assert len(director._pending_birth_requests) == 0
        assert len(director._birth_history) == 1
        assert director._birth_history[0]["zone"] == "lighthouse_bluffs"
    finally:
        restore()
        shutil.rmtree(tmp_world, ignore_errors=True)
    print("  [PASS] queue_birth_request_queues_and_drains")


def test_birth_name_avoids_collision_with_existing_npcs():
    """The name generator should never produce a name that collides
    with an existing NPC. Over multiple births, all names must be
    unique."""
    import tempfile, shutil
    restore = _isolate_state_file("birth_names")
    tmp_world = Path(tempfile.mkdtemp(prefix="test_bn_"))
    (tmp_world / "npc_profiles").mkdir()
    try:
        engine = _make_stub_engine(profile_specs=_ZONED_PROFILES)
        engine.config = SimpleNamespace(
            world_name="TestWorld", world_dir=str(tmp_world),
        )
        director = StoryDirector(engine)
        director._zone_config = {
            "dock_district": {
                "role_pool": ["worker"],
                "lore_hook": "test",
            },
        }
        names = set()
        for _ in range(10):
            request = {"zone": "dock_district", "role": "worker", "reason": "test"}
            result = director._dispatch_npc_birth(request)
            assert result["ok"] is True, result
            name = result["record"]["name"]
            assert name not in names, f"duplicate name: {name}"
            names.add(name)
    finally:
        restore()
        shutil.rmtree(tmp_world, ignore_errors=True)
    print("  [PASS] birth_name_avoids_collision_with_existing_npcs")


# ── Autonomous lifecycle tests (Phase 2c) ──────────────────────

def test_autonomous_off_no_director_deaths():
    """When autonomous lifecycle is off (default), the Director must
    never propose deaths on its own, even if arcs are at confront."""
    engine, director, restore = _seed_zoned_director("auto_off")
    try:
        from npc_engine.story_director import NarrativeArc
        arc = NarrativeArc(
            id="test_arc", theme="Final showdown",
            focus_npcs=["reva"], beat_goals=["s", "e", "confront", "resolve"],
            current_beat=2, status="active", started_at_tick=1,
        )
        director.arc_planner.arcs.append(arc)
        director.arc_planner.active_arc_ids.append("test_arc")
        # autonomous is OFF (default)
        assert director._autonomous_lifecycle is False
        proposal = director._propose_autonomous_death()
        assert proposal is None
    finally:
        restore()
    print("  [PASS] autonomous_off_no_director_deaths")


def test_autonomous_on_proposes_death_at_confront():
    """When autonomous lifecycle is ON and an arc is at confront,
    the Director should propose killing a cast member to resolve
    the arc narratively."""
    engine, director, restore = _seed_zoned_director("auto_on_death")
    try:
        from npc_engine.story_director import NarrativeArc
        arc = NarrativeArc(
            id="arc_kill", theme="Reva's reckoning",
            focus_npcs=["reva", "finn"],
            beat_goals=["seed", "escalate", "confront", "resolve"],
            current_beat=2, status="active", started_at_tick=1,
        )
        director.arc_planner.arcs.append(arc)
        director.arc_planner.active_arc_ids.append("arc_kill")
        director.set_autonomous_lifecycle(True)
        proposal = director._propose_autonomous_death()
        assert proposal is not None, "expected a death proposal"
        assert proposal["npc_id"] in ["reva", "finn"]
        assert "narrative resolution" in proposal["cause"].lower()
    finally:
        restore()
    print("  [PASS] autonomous_on_proposes_death_at_confront")


def test_autonomous_death_capped_by_session_limit():
    """The Director cannot exceed _MAX_AUTONOMOUS_DEATHS_PER_SESSION
    deaths in a single session, even if arcs keep hitting confront."""
    import npc_engine.story_director as sd_mod
    engine, director, restore = _seed_zoned_director("auto_cap")
    try:
        director.set_autonomous_lifecycle(True)
        # Exhaust the cap
        director._autonomous_deaths_this_session = sd_mod._MAX_AUTONOMOUS_DEATHS_PER_SESSION
        from npc_engine.story_director import NarrativeArc
        arc = NarrativeArc(
            id="arc_capped", theme="Another showdown",
            focus_npcs=["reva"],
            beat_goals=["s", "e", "confront", "resolve"],
            current_beat=2, status="active", started_at_tick=1,
        )
        director.arc_planner.arcs.append(arc)
        director.arc_planner.active_arc_ids.append("arc_capped")
        proposal = director._propose_autonomous_death()
        assert proposal is None, "should be capped"
    finally:
        restore()
    print("  [PASS] autonomous_death_capped_by_session_limit")


def test_player_killed_npc_full_propagation():
    """End-to-end scenario: player kills old_bones in the tavern.
    Game client records the player action with trust_delta then
    queues the death with cause='killed by the player...'. Verify
    the Director propagates the death through every surface:
    deceased record, player action trail, quest cleanup, focus
    exclusion, snapshot 'RECENTLY DEPARTED', FactLedger gossip
    entry, and the cause string containing 'player'."""
    restore = _isolate_state_file("player_kill")
    try:
        engine = _make_stub_engine(
            profile_specs=_ZONED_PROFILES,
            responses=[
                # 5 response slots — one per tick after the death
                '{"action": "event", "target": "finn", "event": "finn hears news of the murder"}',
                '{"action": "fact", "npc_id": "finn", "fact": "Old Bones was killed in the tavern last night"}',
                '{"action": "event", "target": "varro", "event": "varro looks over his shoulder warily"}',
                '{"action": "fact", "npc_id": "varro", "fact": "the tavern is closed since Old Bones died"}',
                '{"action": "event", "target": "captain_reva", "event": "reva investigates the scene"}',
            ],
        )
        director = StoryDirector(engine)
        director.set_active_zones(["dock_district"])

        # Give old_bones an active quest that the player had accepted
        from npc_engine.knowledge import Quest
        engine.pie.npc_knowledge.profiles["old_bones"].add_quest(
            Quest(id="tavern_smuggler", name="The Loose Tongue",
                   description="Find the informant", status="active"),
        )
        pre_active_quests = [
            q for q in engine.pie.npc_knowledge.profiles["old_bones"].quests
            if q.status in ("available", "active")
        ]
        assert len(pre_active_quests) == 1

        # 1. Player records the murder action (trust tanks on target)
        player_result = director.record_player_action(
            text="Player drew a blade and killed Old Bones in the tavern",
            target="old_bones",
            trust_delta=-60,
        )
        assert player_result["ok"] is True
        assert director.recent_player_actions[-1]["target"] == "old_bones"

        # 2. Game client queues the death
        death_result = director.queue_death_request(
            npc_id="old_bones",
            cause="killed by the player in the tavern",
            transfers_quests_to=None,
        )
        assert death_result["ok"] is True

        # 3. Next tick drains the death via lifecycle phase
        director.tick()

        # Post-conditions
        old_bones_npc = engine.pie.npc_knowledge.profiles["old_bones"]
        assert old_bones_npc.status == "deceased"
        assert "player" in old_bones_npc.death_cause.lower()
        assert "old_bones" in director._deceased_npcs
        rec = director._deceased_npcs["old_bones"]
        assert "player" in rec["death_cause"].lower()
        assert rec["death_tick"] > 0

        # Quest was aborted (no inheritor given)
        post_quests = [
            q for q in engine.pie.npc_knowledge.profiles["old_bones"].quests
            if q.status in ("available", "active")
        ]
        assert len(post_quests) == 0, (
            "quest should be aborted after giver killed by player"
        )
        assert any(q.status == "aborted" for q in old_bones_npc.quests)

        # Player action trail still carries the murder record so the
        # Director's world snapshot surfaces it as player behavior
        # on subsequent ticks
        assert any(
            "killed old bones" in pa.get("text", "").lower()
            for pa in director.recent_player_actions
        )

        # 4. Future focus picks must exclude old_bones
        for _ in range(10):
            focus = director._pick_focus_npc()
            assert focus != "old_bones", f"deceased picked: {focus}"

        # 5. Snapshot surfaces the death as RECENTLY DEPARTED with
        # the player-specific cause so the LLM sees it as world state
        snap = director._world_snapshot()
        assert "RECENTLY DEPARTED" in snap, snap
        assert "killed by the player" in snap.lower()

        # 6. FactLedger has a death entry for gossip propagation
        ledger_texts = [
            str(e.get("text", "")).lower()
            for e in director.ledger.entries
        ]
        assert any(
            ("dead" in t or "died" in t)
            and ("old bones" in t or "old_bones" in t)
            for t in ledger_texts
        ), f"no FactLedger death entry found: {ledger_texts}"

        # 7. Subsequent ticks continue to dispatch to other NPCs
        # (world isn't frozen just because one NPC died)
        director.tick()
        director.tick()
        # Should have touched various remaining dock NPCs
        remaining_touches = set()
        for d in director.recent_decisions:
            for key in ("action", "sub_actions"):
                val = d.get(key)
                if isinstance(val, dict):
                    nid = val.get("npc_id") or val.get("target")
                    if nid and nid != "old_bones":
                        remaining_touches.add(nid)
                elif isinstance(val, list):
                    for sub in val:
                        a = sub.get("action", {}) if isinstance(sub, dict) else {}
                        nid = a.get("npc_id") or a.get("target")
                        if nid and nid != "old_bones":
                            remaining_touches.add(nid)
        assert len(remaining_touches) >= 1, (
            f"rotation should continue post-death, touched: {remaining_touches}"
        )
    finally:
        restore()
    print("  [PASS] player_killed_npc_full_propagation")


def test_autonomous_skips_death_when_no_arc_at_confront():
    """When autonomous is ON but no arc is at confront beat, the
    Director should NOT propose any death."""
    engine, director, restore = _seed_zoned_director("auto_no_confront")
    try:
        from npc_engine.story_director import NarrativeArc
        arc = NarrativeArc(
            id="arc_early", theme="Just started",
            focus_npcs=["reva"],
            beat_goals=["seed", "escalate", "confront", "resolve"],
            current_beat=0, status="active", started_at_tick=1,
        )
        director.arc_planner.arcs.append(arc)
        director.arc_planner.active_arc_ids.append("arc_early")
        director.set_autonomous_lifecycle(True)
        proposal = director._propose_autonomous_death()
        assert proposal is None
    finally:
        restore()
    print("  [PASS] autonomous_skips_death_when_no_arc_at_confront")


def test_pending_player_target_bypasses_zone_filter():
    """When the player has an outstanding action targeting an
    out-of-zone NPC (e.g. sent a letter ahead), that NPC must still
    win focus selection — player reactivity is layer 1, zone filter
    is layer 2."""
    restore = _isolate_state_file("zone_player_bypass")
    try:
        engine = _make_stub_engine(profile_specs=_ZONED_PROFILES)
        director = StoryDirector(engine)
        director.set_active_zones(["dock_district"])
        # Player's action targets thessa (out of active zone).
        director.recent_player_actions = [{
            "at": "2026-04-15T10:05:00+00:00",
            "tick_at_time": 1,
            "text": "Player sent a messenger to the hermit",
            "target": "thessa",
        }]
        focus = director._pick_focus_npc()
        # Player reactivity wins over zone filter.
        assert focus == "thessa", focus
    finally:
        restore()
    print("  [PASS] pending_player_target_bypasses_zone_filter")


def test_single_action_tick_is_backward_compatible():
    """tick() default behavior must still return the legacy shape with
    'action' and 'dispatch' at the top level — existing clients depend
    on it."""
    restore = _isolate_state_file("single_compat")
    try:
        engine = _make_stub_engine(responses=[
            '{"action": "event", "target": "kael", "event": "x"}',
        ])
        director = StoryDirector(engine)
        result = director.tick()  # actions_per_tick default = 1
        assert "action" in result
        assert "dispatch" in result
        assert "raw_response" in result
        assert "sub_actions" not in result
    finally:
        restore()
    print("  [PASS] single_action_tick_is_backward_compatible")


def test_dialogue_autofeed_format():
    """The engine.process wiring formats player dialogue as 'Player said
    to <npc_id>: <text>' so the Director sees both speaker intent and
    listener in one record."""
    restore = _isolate_state_file("dialogue_format")
    try:
        engine = _make_stub_engine()
        director = StoryDirector(engine)
        result = director.record_player_action(
            text="Player said to noah: What does the letter say?",
            target="noah",
        )
        assert result["ok"]
        snap = director.recent_player_actions[0]
        assert snap["text"].startswith("Player said to noah:")
        assert snap["target"] == "noah"
    finally:
        restore()
    print("  [PASS] dialogue_autofeed_format")


def test_record_player_action_and_snapshot():
    restore = _isolate_state_file("player")
    try:
        engine = _make_stub_engine()
        director = StoryDirector(engine)
        result = director.record_player_action(
            "Player publicly accused Mara of selling counterfeit steel",
            target="mara",
        )
        assert result["ok"], result
        assert len(director.recent_player_actions) == 1
        snapshot = director._world_snapshot()
        assert "PLAYER:" in snapshot, snapshot
        assert "accused Mara" in snapshot, snapshot
    finally:
        restore()
    print("  [PASS] record_player_action_and_snapshot")


def test_focus_npc_prioritizes_pending_player_target():
    """Pending player actions must override round-robin rotation so the
    Director actually responds to player moves."""
    restore = _isolate_state_file("pending")
    try:
        engine = _make_stub_engine()
        director = StoryDirector(engine)
        # Set up rotation state — kael was just touched two ticks ago,
        # bess touched recently, noah least recently touched → rotation
        # would pick noah.
        director.tick_count = 3
        director.last_tick_at = "2026-04-13T10:00:00+00:00"
        director.recent_decisions = [
            {"tick": 1, "action": {"action": "event", "target": "kael", "event": "x"}},
            {"tick": 2, "action": {"action": "event", "target": "bess", "event": "y"}},
        ]
        # Without a player action, rotation picks least-recent (noah)
        assert director._pick_focus_npc() == "noah"

        # Record a player action targeting bess AFTER the last tick — it
        # should now win over rotation even though bess was just touched.
        director.recent_player_actions = [{
            "at": "2026-04-13T10:05:00+00:00",
            "tick_at_time": 3,
            "text": "Player threatened Bess",
            "target": "bess",
        }]
        assert director._pick_focus_npc() == "bess"
    finally:
        restore()
    print("  [PASS] focus_npc_prioritizes_pending_player_target")


def test_pending_player_target_ignores_stale_actions():
    """An action older than last_tick_at has already been reacted to and
    should NOT trigger another response."""
    restore = _isolate_state_file("stale")
    try:
        engine = _make_stub_engine()
        director = StoryDirector(engine)
        director.tick_count = 5
        director.last_tick_at = "2026-04-13T11:00:00+00:00"
        director.recent_player_actions = [{
            "at": "2026-04-13T10:00:00+00:00",  # before last tick
            "tick_at_time": 4,
            "text": "Old player action",
            "target": "kael",
        }]
        assert director._pending_player_target(["kael", "bess", "noah"]) is None
    finally:
        restore()
    print("  [PASS] pending_player_target_ignores_stale_actions")


def test_fact_ledger_flags_similar_text():
    """Two fact-shaped strings about the same NPC and topic should
    trigger a similarity warning when added back-to-back."""
    tmp_path = NPC_ROOT / "data" / "story_director" / "_tmp_ledger_sim.json"
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        ledger = FactLedger(tmp_path, threshold=0.6)
        if ledger.embedder is None:
            print("  [SKIP] fact_ledger_flags_similar_text — embedder unavailable")
            return
        first = ledger.add(
            "Mara is hiding a strange package under her shop's floorboards.",
            npc_id="mara", kind="fact", tick=1,
        )
        assert first is None, first  # nothing to compare against
        second = ledger.add(
            "Mara has a hidden package beneath the floorboards in her shop.",
            npc_id="mara", kind="fact", tick=2,
        )
        assert second is not None, "expected similarity warning on near-paraphrase"
        assert second["similarity"] >= 0.6, second
        assert "Mara" in second["matches_text"]
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    print("  [PASS] fact_ledger_flags_similar_text")


def test_fact_ledger_does_not_flag_unrelated_text():
    tmp_path = NPC_ROOT / "data" / "story_director" / "_tmp_ledger_unrel.json"
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        ledger = FactLedger(tmp_path, threshold=0.72)
        if ledger.embedder is None:
            print("  [SKIP] fact_ledger_does_not_flag_unrelated_text — embedder unavailable")
            return
        ledger.add(
            "Kael the blacksmith forged a new sword from rare silver ore.",
            npc_id="kael", kind="fact", tick=1,
        )
        warning = ledger.add(
            "Pip the urchin found a copper coin in the gutter.",
            npc_id="pip", kind="fact", tick=2,
        )
        assert warning is None, f"unexpected warning: {warning}"
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    print("  [PASS] fact_ledger_does_not_flag_unrelated_text")


def test_fact_ledger_persists_and_reloads():
    tmp_path = NPC_ROOT / "data" / "story_director" / "_tmp_ledger_persist.json"
    tmp_emb = tmp_path.with_suffix(".embeddings.npy")
    for p in (tmp_path, tmp_emb):
        if p.exists():
            p.unlink()
    try:
        ledger = FactLedger(tmp_path)
        if ledger.embedder is None:
            print("  [SKIP] fact_ledger_persists_and_reloads — embedder unavailable")
            return
        ledger.add("Bess hears a strange whisper at the inn.",
                   npc_id="bess", kind="event", tick=1)
        assert tmp_path.exists()
        assert tmp_emb.exists(), "embeddings sidecar should be created"
        ledger2 = FactLedger(tmp_path)
        assert len(ledger2.entries) == 1, ledger2.entries
        assert ledger2.entries[0]["npc_id"] == "bess"
        # Embeddings must round-trip so similarity still works after
        # a reload — check the key is present and non-empty
        assert "embedding" in ledger2.entries[0]
        assert len(ledger2.entries[0]["embedding"]) > 0
    finally:
        for p in (tmp_path, tmp_emb):
            if p.exists():
                p.unlink()
    print("  [PASS] fact_ledger_persists_and_reloads")


def test_ledger_sidecar_round_trips_embedding_values():
    """Round-tripping through the binary sidecar must preserve the
    embedding values within float32 precision. This is what lets
    similarity checks stay stable across a save/load cycle."""
    import numpy as np
    tmp_path = NPC_ROOT / "data" / "story_director" / "_tmp_ledger_roundtrip.json"
    tmp_emb = tmp_path.with_suffix(".embeddings.npy")
    for p in (tmp_path, tmp_emb):
        if p.exists():
            p.unlink()
    try:
        ledger = FactLedger(tmp_path)
        ledger._np = np
        ledger._embedder = False  # skip real embedder
        # Stuff hand-crafted entries directly
        ledger.entries = [
            {"text": "scene A",
             "embedding": [0.10, 0.20, 0.30, 0.40],
             "npc_id": "bess", "kind": "event", "tick": 1},
            {"text": "scene B",
             "embedding": [0.50, 0.60, 0.70, 0.80],
             "npc_id": "kael", "kind": "fact", "tick": 2},
        ]
        ledger._save()
        assert tmp_path.exists()
        assert tmp_emb.exists()

        # The JSON should NOT contain the embedding field anymore
        raw = json.loads(tmp_path.read_text(encoding="utf-8"))
        for e in raw["entries"]:
            assert "embedding" not in e, (
                "JSON metadata should not carry embeddings after compression"
            )

        # Reload and verify embeddings came back (within float32 eps)
        ledger2 = FactLedger(tmp_path)
        assert len(ledger2.entries) == 2
        for original, reloaded in zip(ledger.entries, ledger2.entries):
            assert reloaded["text"] == original["text"]
            assert reloaded["npc_id"] == original["npc_id"]
            for a, b in zip(original["embedding"], reloaded["embedding"]):
                assert abs(a - b) < 1e-5, f"{a} vs {b}"
    finally:
        for p in (tmp_path, tmp_emb):
            if p.exists():
                p.unlink()
    print("  [PASS] ledger_sidecar_round_trips_embedding_values")


def test_ledger_loads_legacy_inline_format():
    """Old fact_ledger.json files (pre-compression) had embeddings
    inline in the JSON. Those must still load cleanly so existing
    state files don't need migration."""
    tmp_path = NPC_ROOT / "data" / "story_director" / "_tmp_ledger_legacy.json"
    tmp_emb = tmp_path.with_suffix(".embeddings.npy")
    for p in (tmp_path, tmp_emb):
        if p.exists():
            p.unlink()
    try:
        # Hand-write an old-format JSON file (embeddings inline, no sidecar)
        legacy_data = {
            "entries": [
                {"text": "legacy entry",
                 "embedding": [0.1, 0.2, 0.3, 0.4],
                 "npc_id": "bess", "kind": "event", "tick": 1},
            ],
        }
        tmp_path.write_text(json.dumps(legacy_data), encoding="utf-8")

        ledger = FactLedger(tmp_path)
        assert len(ledger.entries) == 1
        assert ledger.entries[0]["embedding"] == [0.1, 0.2, 0.3, 0.4]
        assert ledger.entries[0]["npc_id"] == "bess"
    finally:
        for p in (tmp_path, tmp_emb):
            if p.exists():
                p.unlink()
    print("  [PASS] ledger_loads_legacy_inline_format")


def test_ledger_sidecar_is_smaller_than_inline_json():
    """Binary float32 sidecar must be meaningfully smaller than the
    same data stored as JSON float64 lists — that's the whole point
    of compression. Test a realistic embedding size (384 floats)."""
    import numpy as np
    tmp_path = NPC_ROOT / "data" / "story_director" / "_tmp_ledger_size.json"
    tmp_emb = tmp_path.with_suffix(".embeddings.npy")
    for p in (tmp_path, tmp_emb):
        if p.exists():
            p.unlink()
    try:
        ledger = FactLedger(tmp_path)
        ledger._np = np
        ledger._embedder = False
        # 10 entries with 384-float embeddings (typical size)
        rng = np.random.RandomState(42)
        ledger.entries = [
            {"text": f"entry {i}",
             "embedding": rng.randn(384).astype(float).tolist(),
             "npc_id": "bess", "kind": "event", "tick": i}
            for i in range(10)
        ]
        ledger._save()

        json_size = tmp_path.stat().st_size
        emb_size = tmp_emb.stat().st_size
        total_new = json_size + emb_size

        # What the old format would have been: serialize entries WITH
        # embeddings inline and measure
        legacy_data = {"entries": ledger.entries}
        legacy_bytes = json.dumps(legacy_data, indent=2).encode("utf-8")
        legacy_size = len(legacy_bytes)

        # Expect the new format to be at least 5x smaller than inline
        # (embeddings dominate, float64-as-JSON is ~20 bytes/float,
        # binary float32 is 4 bytes/float → ~5x best case)
        ratio = legacy_size / total_new
        assert ratio >= 4.0, (
            f"expected >=4x reduction, got {ratio:.2f}x "
            f"(legacy={legacy_size}, new={total_new})"
        )
    finally:
        for p in (tmp_path, tmp_emb):
            if p.exists():
                p.unlink()
    print("  [PASS] ledger_sidecar_is_smaller_than_inline_json")


def test_contradiction_checker_catches_known_contradiction():
    """A clear premise/hypothesis contradiction should be classified
    as 'contradiction' by the small NLI model. This is the model's job
    — if this fails, the model is broken, not our wrapper."""
    checker = ContradictionChecker()
    if checker.model is None:
        print("  [SKIP] contradiction_checker_catches_known_contradiction — model unavailable")
        return
    result = checker.check(
        premise="Mara is hiding contraband under the floorboards of her shop.",
        hypothesis="Mara has nothing hidden in her shop. The accusations are false.",
    )
    assert result is not None
    assert result["label"] == "contradiction", result
    assert result["is_contradiction"] is True, result
    assert result["confidence"] >= 0.55, result
    print("  [PASS] contradiction_checker_catches_known_contradiction")


def test_contradiction_checker_does_not_flag_paraphrase():
    """A near-paraphrase pair (same fact, different wording) should
    NOT be classified as contradiction. The small NLI model is
    hypersensitive on topically-related-but-distinct pairs, so the
    threshold at 0.85 is what filters those false positives."""
    checker = ContradictionChecker()
    if checker.model is None:
        print("  [SKIP] contradiction_checker_does_not_flag_paraphrase — model unavailable")
        return
    result = checker.check(
        premise="Mara is a merchant who runs a fabric shop in Ashenvale.",
        hypothesis="Mara owns a fabric shop in the village of Ashenvale and works as a merchant.",
    )
    assert result is not None, result
    assert result["is_contradiction"] is False, result
    print("  [PASS] contradiction_checker_does_not_flag_paraphrase")


def test_contradiction_checker_does_not_flag_plot_escalation():
    """A plot escalation (fact B builds on fact A without contradicting
    it) should also not be flagged. This is the most common Director
    output pattern and the most important false-positive case to
    eliminate."""
    checker = ContradictionChecker()
    if checker.model is None:
        print("  [SKIP] contradiction_checker_does_not_flag_plot_escalation — model unavailable")
        return
    result = checker.check(
        premise="Noah the elder is troubled by an old letter from the king.",
        hypothesis="Noah the elder confided in the player that the king's letter weighs heavily on him.",
    )
    assert result is not None, result
    # The pair may register as 'entailment' or 'neutral' — both are fine,
    # we only care that it's not flagged as is_contradiction at our
    # threshold (0.85).
    assert result["is_contradiction"] is False, result
    print("  [PASS] contradiction_checker_does_not_flag_plot_escalation")


def test_contradiction_checker_silent_when_unavailable():
    checker = ContradictionChecker()
    checker._model = False  # force-disable
    result = checker.check("a", "b")
    assert result is None
    print("  [PASS] contradiction_checker_silent_when_unavailable")


def test_fact_ledger_warning_includes_nli_when_model_available():
    """When the NLI model is loaded, ledger warnings should include an
    nli block with label, confidence, and per-class scores."""
    tmp_path = NPC_ROOT / "data" / "story_director" / "_tmp_ledger_nli.json"
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        ledger = FactLedger(tmp_path, threshold=0.5)
        if ledger.embedder is None:
            print("  [SKIP] fact_ledger_warning_includes_nli_when_model_available — embedder unavailable")
            return
        if ledger.contradiction_checker.model is None:
            print("  [SKIP] fact_ledger_warning_includes_nli_when_model_available — NLI unavailable")
            return
        ledger.add(
            "Mara is openly hiding contraband under the floorboards of her shop.",
            npc_id="mara", kind="fact", tick=1,
        )
        warning = ledger.add(
            "Mara has nothing hidden in her shop and the accusations are false.",
            npc_id="mara", kind="fact", tick=2,
        )
        assert warning is not None, "expected a similarity match"
        assert "nli" in warning, warning
        assert warning["nli"]["label"] in ContradictionChecker.LABELS
        # The pair is a clear contradiction; should be flagged
        assert warning.get("contradiction") is True, warning
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    print("  [PASS] fact_ledger_warning_includes_nli_when_model_available")


def test_contradiction_retry_redoes_worker_when_pre_check_fires():
    """Pre-seed the ledger with a fact, then run a worker whose FIRST
    LLM response contradicts that fact. The retry path should kick in
    and the SECOND response should be the one dispatched."""
    restore = _isolate_state_file("retry")
    try:
        engine = _make_stub_engine(responses=[
            # First response: directly contradicts the seeded fact
            '{"action": "fact", "npc_id": "mara", '
            '"fact": "Mara has nothing hidden in her shop. The accusations are completely false."}',
            # Second response (after retry): non-contradicting alternative
            '{"action": "fact", "npc_id": "mara", '
            '"fact": "Mara is reorganizing her fabric inventory after a busy week."}',
        ])
        director = StoryDirector(engine)
        if director.ledger.embedder is None or director.ledger.contradiction_checker.model is None:
            print("  [SKIP] contradiction_retry_redoes_worker_when_pre_check_fires — model unavailable")
            return

        # Seed the ledger with a fact the first response will contradict
        director.ledger.add(
            text="Mara is openly hiding contraband under the floorboards of her shop.",
            npc_id="mara", kind="fact", tick=0,
        )

        # Pin the rotation so this tick fires a fact for Mara.
        director._kind_rotation_index = _ACTION_KIND_ROTATION_INDEX_FOR("fact")
        # Force focus to mara directly via player action so reactivity wins
        director.recent_player_actions = [{
            "at": "2030-01-01T00:00:00+00:00",
            "tick_at_time": 0,
            "text": "Player asked Mara about the rumors.",
            "target": "mara",
        }]

        result = director.tick()
        # Both responses were consumed (one for first attempt, one for retry)
        assert len(engine.pie.base_model.prompts) == 2, engine.pie.base_model.prompts
        # The dispatched action is the SECOND response — non-contradicting
        assert "reorganizing" in str(result["action"]), result
        # The dispatch result is flagged as retried
        assert result["dispatch"].get("retried_after_contradiction") is True, result["dispatch"]
    finally:
        restore()
    print("  [PASS] contradiction_retry_redoes_worker_when_pre_check_fires")


def test_no_retry_when_pre_check_finds_no_contradiction():
    """If the pre-check returns None (no contradiction), the worker
    should dispatch the FIRST response — no retry, single LLM call."""
    restore = _isolate_state_file("no_retry")
    try:
        engine = _make_stub_engine(responses=[
            '{"action": "fact", "npc_id": "noah", '
            '"fact": "Noah has been spending more time in his study lately."}',
        ])
        director = StoryDirector(engine)
        if director.ledger.embedder is None:
            print("  [SKIP] no_retry_when_pre_check_finds_no_contradiction — embedder unavailable")
            return
        director._kind_rotation_index = _ACTION_KIND_ROTATION_INDEX_FOR("fact")
        director.recent_player_actions = [{
            "at": "2030-01-01T00:00:00+00:00",
            "tick_at_time": 0,
            "text": "Player visited Noah.",
            "target": "noah",
        }]
        result = director.tick()
        assert len(engine.pie.base_model.prompts) == 1, "expected exactly one LLM call"
        assert result["dispatch"].get("retried_after_contradiction") is not True, result["dispatch"]
    finally:
        restore()
    print("  [PASS] no_retry_when_pre_check_finds_no_contradiction")


def test_fact_ledger_check_separates_from_add():
    """The check() method must compute a warning without storing the
    candidate in the ledger — that's what enables pre-dispatch retry."""
    tmp_path = NPC_ROOT / "data" / "story_director" / "_tmp_ledger_check.json"
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        ledger = FactLedger(tmp_path, threshold=0.5)
        if ledger.embedder is None:
            print("  [SKIP] fact_ledger_check_separates_from_add — embedder unavailable")
            return
        ledger.add("Mara is hiding contraband under the floorboards.",
                   npc_id="mara", kind="fact", tick=1)
        # check() should find the prior entry but NOT add the new candidate
        warning = ledger.check("Mara has a hidden package beneath the floorboards.")
        assert warning is not None
        assert len(ledger.entries) == 1, "check() should not store the candidate"
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    print("  [PASS] fact_ledger_check_separates_from_add")


def test_fact_ledger_silent_when_embedder_missing():
    """If sentence-transformers is unavailable, add() must return None
    silently — the Director still works without the ledger."""
    tmp_path = NPC_ROOT / "data" / "story_director" / "_tmp_ledger_miss.json"
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        ledger = FactLedger(tmp_path)
        ledger._embedder = False  # force-disable
        warning = ledger.add("any text", npc_id="x", kind="fact", tick=1)
        assert warning is None
        assert ledger.entries == []
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    print("  [PASS] fact_ledger_silent_when_embedder_missing")


def test_record_player_action_routes_quest_completion_to_engine():
    """Setting quest_completed should call engine.complete_quest and
    record the result in the player action."""
    restore = _isolate_state_file("quest_complete")
    try:
        engine = _make_stub_engine()
        director = StoryDirector(engine)
        result = director.record_player_action(
            text="Player returned the stolen hammers to Kael.",
            target="kael",
            trust_delta=10,
            quest_completed="missing_hammers",
        )
        assert result["ok"], result
        assert ("complete_quest", "missing_hammers") in engine.calls
        assert ("adjust_trust", "kael", 10, "player: Player returned the stolen hammers to Kael.") in engine.calls
        # Player quest tracker now reflects the completion
        assert any(q["id"] == "missing_hammers"
                   for q in engine.pie.player_quests.completed_quests)
        # Record carries the completion marker
        last = director.recent_player_actions[-1]
        assert last.get("quest_completed") == "missing_hammers", last
    finally:
        restore()
    print("  [PASS] record_player_action_routes_quest_completion_to_engine")


def test_record_player_action_routes_quest_acceptance_to_engine():
    restore = _isolate_state_file("quest_accept")
    try:
        engine = _make_stub_engine()
        director = StoryDirector(engine)
        result = director.record_player_action(
            text="Player accepted Bess's tavern rumors quest.",
            target="bess",
            quest_accepted={
                "id": "tavern_rumors",
                "name": "Tavern Rumors",
                "given_by": "bess",
            },
        )
        assert result["ok"], result
        assert ("accept_quest", "tavern_rumors", "Tavern Rumors", "bess") in engine.calls
        assert any(q["id"] == "tavern_rumors"
                   for q in engine.pie.player_quests.active_quests)
        last = director.recent_player_actions[-1]
        assert last.get("quest_accepted") == "tavern_rumors", last
    finally:
        restore()
    print("  [PASS] record_player_action_routes_quest_acceptance_to_engine")


def test_player_actions_trimmed_to_last_8():
    restore = _isolate_state_file("player_trim")
    try:
        engine = _make_stub_engine()
        director = StoryDirector(engine)
        for i in range(12):
            director.record_player_action(f"player action {i}")
        assert len(director.recent_player_actions) == 8
        assert director.recent_player_actions[0]["text"] == "player action 4", \
            director.recent_player_actions[0]
    finally:
        restore()
    print("  [PASS] player_actions_trimmed_to_last_8")


def test_parse_noop_strips_hallucinated_fields():
    """0.5B regression — noops with dummy quest/npc_id fields should be
    coerced to a clean ``{action, reason}`` shape."""
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    raw = (
        '{"action": "noop", "reason": "nothing to do", '
        '"npc_id": "none", "quest": {"id": "none", "name": "none"}}'
    )
    action = director._parse_action(raw)
    assert action == {"action": "noop", "reason": "nothing to do"}, action
    print("  [PASS] parse_noop_strips_hallucinated_fields")


def test_dispatch_quest_adds_to_npc():
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    action = {
        "action": "quest",
        "npc_id": "kael",
        "quest": {
            "id": "missing_hammers",
            "name": "The Missing Hammers",
            "description": "Find the thief.",
            "reward": "A forged blade.",
            "objectives": ["Investigate the forge at night."],
        },
    }
    result = director._dispatch(action)
    assert result["ok"], result
    kael = engine.pie.npc_knowledge.get("kael")
    assert len(kael.quests) == 1, kael.quests
    assert kael.quests[0].id == "missing_hammers"
    assert kael.quests[0].objectives == ["Investigate the forge at night."]
    # Quest dispatch should also emit a global announcement event
    assert any("new work" in e.description.lower() or "missing hammers" in e.description.lower()
               for e in kael.events), [e.description for e in kael.events]
    print("  [PASS] dispatch_quest_adds_to_npc")


def test_dispatch_quest_dedupes_by_id():
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    action = {
        "action": "quest",
        "npc_id": "kael",
        "quest": {"id": "dup", "name": "Dup", "description": "x"},
    }
    first = director._dispatch(action)
    second = director._dispatch(action)
    assert first["ok"], first
    assert not second["ok"], second
    assert "already_exists" in second["reason"], second
    print("  [PASS] dispatch_quest_dedupes_by_id")


def test_dispatch_event_global_vs_targeted():
    engine = _make_stub_engine()
    director = StoryDirector(engine)

    global_action = {"action": "event", "target": "all", "event": "the sky turned red"}
    result = director._dispatch(global_action)
    assert result["ok"], result
    # Every NPC saw it
    for npc in engine.pie.npc_knowledge.profiles.values():
        assert any("sky turned red" in e.description for e in npc.events)

    targeted = {"action": "event", "target": "bess", "event": "a stranger entered the inn"}
    result = director._dispatch(targeted)
    assert result["ok"], result
    bess = engine.pie.npc_knowledge.get("bess")
    assert any("stranger entered" in e.description for e in bess.events)

    bad = {"action": "event", "target": "nobody", "event": "x"}
    result = director._dispatch(bad)
    assert not result["ok"], result
    assert "unknown_target" in result["reason"], result
    print("  [PASS] dispatch_event_global_vs_targeted")


def test_dispatch_fact_world_and_personal():
    engine = _make_stub_engine()
    director = StoryDirector(engine)

    world_fact = {"action": "fact", "npc_id": "noah",
                  "fact": "The tax collector was found dead.", "fact_type": "world"}
    result = director._dispatch(world_fact)
    assert result["ok"], result

    personal_fact = {"action": "fact", "npc_id": "noah",
                     "fact": "I fear the king's letter.", "fact_type": "personal"}
    result = director._dispatch(personal_fact)
    assert result["ok"], result

    noah = engine.pie.npc_knowledge.get("noah")
    # Director-injected facts route into the dynamic lanes, not the
    # static profile lists. See engine.add_knowledge in engine.py.
    assert "The tax collector was found dead." in noah.dynamic_world_facts
    assert "I fear the king's letter." in noah.dynamic_personal_knowledge
    assert "The tax collector was found dead." not in noah.world_facts
    assert "I fear the king's letter." not in noah.personal_knowledge
    print("  [PASS] dispatch_fact_world_and_personal")


def test_full_tick_loop_with_stubbed_model():
    restore = _isolate_state_file("full_tick")
    try:
        engine = _make_stub_engine(responses=[
            '{"action": "quest", "reason": "kael tension", "npc_id": "kael", '
            '"quest": {"id": "tick_quest", "name": "Test Quest", '
            '"description": "Go look.", "reward": "thanks", "objectives": ["Look"]}}'
        ])
        director = StoryDirector(engine)
        # Pin rotation to 'quest' so the Python-side kind forcing agrees
        # with what the stub model emits.
        director._kind_rotation_index = _ACTION_KIND_ROTATION_INDEX_FOR("quest")
        result = director.tick()
        assert result["action"]["action"] == "quest", result
        assert result["dispatch"]["ok"], result
        assert director.tick_count == 1
        assert len(director.recent_decisions) == 1
        # focus NPC will be the first least-recently-touched profile — our
        # stub has kael, bess, noah in that insertion order → kael is focus.
        kael = engine.pie.npc_knowledge.get("kael")
        assert any(q.id == "tick_quest" for q in kael.quests), kael.quests
        assert engine.pie.base_model.prompts, "expected base model to be called"
        prompt = engine.pie.base_model.prompts[0]
        assert "kael" in prompt and "bess" in prompt and "noah" in prompt, prompt[:500]
    finally:
        restore()
    print("  [PASS] full_tick_loop_with_stubbed_model")


def test_tick_persists_state_between_instances():
    restore = _isolate_state_file("persist")
    try:
        engine = _make_stub_engine(responses=[
            '{"action": "noop", "reason": "quiet"}',
        ])
        director = StoryDirector(engine)
        director.tick()
        assert director.tick_count == 1

        # Fresh director should load state from disk
        engine2 = _make_stub_engine(responses=[])
        director2 = StoryDirector(engine2)
        assert director2.tick_count == 1, director2.tick_count
        assert director2.recent_decisions, director2.recent_decisions
    finally:
        restore()
    print("  [PASS] tick_persists_state_between_instances")


def test_cross_session_persistence_smoke():
    """
    End-to-end persistence: run a few ticks, simulate engine shutdown,
    boot a fresh director against the same state files, verify the
    second director picked up tick_count, recent_decisions, and ledger
    entries from the first session. Unit-level test (stub engine) —
    the real-engine version is in the integration smoke test above.
    """
    restore = _isolate_state_file("persist_smoke")
    try:
        # Session 1 — three ticks
        engine1 = _make_stub_engine(responses=[
            '{"action": "event", "target": "kael", "event": "Kael forged a new blade."}',
            '{"action": "event", "target": "bess", "event": "Bess heard a traveler arrive."}',
            '{"action": "event", "target": "noah", "event": "Noah considered the king\'s letter."}',
        ])
        director1 = StoryDirector(engine1)
        director1.tick()
        director1.tick()
        director1.tick()
        assert director1.tick_count == 3
        assert len(director1.recent_decisions) == 3
        first_session_ledger_count = len(director1.ledger.entries)

        # Session 2 — fresh engine, same state files
        engine2 = _make_stub_engine(responses=[
            '{"action": "event", "target": "elara", "event": "Elara returned at dawn."}',
        ])
        director2 = StoryDirector(engine2)
        assert director2.tick_count == 3, director2.tick_count
        assert len(director2.recent_decisions) == 3, director2.recent_decisions
        assert len(director2.ledger.entries) == first_session_ledger_count, \
            (len(director2.ledger.entries), first_session_ledger_count)

        # One more tick should advance the count to 4
        director2.tick()
        assert director2.tick_count == 4

        # Third reload to confirm persistence still works after multi sessions
        engine3 = _make_stub_engine(responses=[])
        director3 = StoryDirector(engine3)
        assert director3.tick_count == 4, director3.tick_count
    finally:
        restore()
    print("  [PASS] cross_session_persistence_smoke")


# ── Integration smoke test (optional) ──────────────────────────

def _pie_model_available() -> bool:
    model = PIE_ROOT / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    return model.exists()


def test_integration_tick_mutates_world():
    if not _pie_model_available():
        print("  [SKIP] integration_tick_mutates_world — PIE model file not found")
        return

    import yaml
    raw = yaml.safe_load((PIE_ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["base_model"]["path"] = str(PIE_ROOT / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf")
    raw["base_model"]["context_length"] = 4096
    raw["base_model"]["temperature"] = 0.6
    raw["fusion"] = raw.get("fusion") or {}
    raw["fusion"]["chat_format"] = "chatml"
    raw["npc"] = raw.get("npc") or {}
    raw["npc"]["enabled"] = True
    raw["npc"]["profiles_dir"] = str(NPC_ROOT / "data" / "worlds" / "ashenvale" / "npc_profiles")
    raw["npc"]["state_dir"] = str(NPC_ROOT / "data" / "worlds" / "ashenvale" / "npc_state")

    temp_pie = PIE_ROOT / "config_test_story.yaml"
    temp_pie.write_text(yaml.dump(raw, default_flow_style=False), encoding="utf-8")
    npc_cfg = {
        "world_dir": str(NPC_ROOT / "data" / "worlds" / "ashenvale"),
        "world_name": "Ashenvale",
        "active_npc": "noah",
        "pie_config": str(temp_pie),
    }
    temp_npc = NPC_ROOT / "config_test_story.yaml"
    temp_npc.write_text(yaml.dump(npc_cfg, default_flow_style=False), encoding="utf-8")

    os.chdir(PIE_ROOT)
    try:
        from engine.license import LicenseState
        LicenseState.reset()
    except Exception:
        pass

    from npc_engine.engine import NPCEngine
    engine = NPCEngine(str(temp_npc))
    engine.initialize()
    assert engine.story_director is not None, "Story Director should be initialized"

    # Reset the ledger so prior bench/test runs don't pollute the
    # integration check with stale entries.
    engine.story_director.ledger.reset()

    # Dialogue auto-feed: a real engine.process call should cause the
    # Director to record the player's input as a player action.
    pre_player_actions = len(engine.story_director.recent_player_actions)
    _ = engine.process("Greetings, elder. What troubles the village?", npc_id="noah")
    post_player_actions = len(engine.story_director.recent_player_actions)
    assert post_player_actions == pre_player_actions + 1, (
        f"dialogue auto-feed missing — was {pre_player_actions}, now {post_player_actions}"
    )
    last_action = engine.story_director.recent_player_actions[-1]
    assert last_action["target"] == "noah", last_action
    assert "noah" in last_action["text"].lower(), last_action

    # Capture pre-state counts
    pre_quests = {npc_id: len(npc.quests)
                  for npc_id, npc in engine.pie.npc_knowledge.profiles.items()}
    pre_events = {npc_id: len(npc.events)
                  for npc_id, npc in engine.pie.npc_knowledge.profiles.items()}
    pre_facts = {npc_id: (len(npc.world_facts) + len(npc.personal_knowledge)
                          + len(npc.dynamic_world_facts)
                          + len(npc.dynamic_personal_knowledge))
                 for npc_id, npc in engine.pie.npc_knowledge.profiles.items()}

    result = engine.story_director.tick(max_tokens=300)
    print(f"    tick result action: {result['action'].get('action')}")
    print(f"    tick result dispatch: {result['dispatch']}")
    print(f"    raw preview: {str(result['raw_response'])[:200]}")

    post_quests = {npc_id: len(npc.quests)
                   for npc_id, npc in engine.pie.npc_knowledge.profiles.items()}
    post_events = {npc_id: len(npc.events)
                   for npc_id, npc in engine.pie.npc_knowledge.profiles.items()}
    post_facts = {npc_id: (len(npc.world_facts) + len(npc.personal_knowledge)
                           + len(npc.dynamic_world_facts)
                           + len(npc.dynamic_personal_knowledge))
                  for npc_id, npc in engine.pie.npc_knowledge.profiles.items()}

    mutated = (
        any(post_quests[k] > pre_quests[k] for k in pre_quests)
        or any(post_events[k] > pre_events[k] for k in pre_events)
        or any(post_facts[k] > pre_facts[k] for k in pre_facts)
    )

    # A parse failure is informative, not fatal — print diagnostics
    if not mutated:
        print("    [WARN] world not mutated — likely parse/dispatch failure")
        print(f"    action: {json.dumps(result['action'], indent=2)}")
        print(f"    dispatch: {result['dispatch']}")

    engine.shutdown()
    try:
        temp_pie.unlink()
        temp_npc.unlink()
    except Exception:
        pass

    assert mutated, "integration tick did not mutate world state"
    print("  [PASS] integration_tick_mutates_world")


# ── Narrative arc tests ─────────────────────────────────────────

def _inject_fake_ledger_entries(ledger: FactLedger, entries: list[dict]):
    """
    Stuff pre-computed entries into a FactLedger for arc tests. Each
    entry needs ``text``, ``npc_id``, ``kind``, ``tick``, and a
    ``embedding`` list — the ArcPlanner reads these fields directly and
    never touches the real embedder, so tests stay fast and deterministic.
    """
    import numpy as np
    ledger._np = np
    ledger._embedder = False  # make sure nobody lazy-loads a real one
    ledger.entries = entries


def _fake_embedding(similarity: float) -> list[float]:
    """
    Return a 2-D unit vector whose dot product with [1, 0] equals
    ``similarity``. Useful for crafting clusters with known cosine
    similarity to a base vector without loading a real embedder.
    """
    import math
    angle = math.acos(max(-1.0, min(1.0, similarity)))
    return [math.cos(angle), math.sin(angle)]


def _base_embedding() -> list[float]:
    return [1.0, 0.0]


def test_arc_planner_proposes_from_clustered_ledger():
    """A ledger with 3 mara-themed entries (high similarity) and 1
    unrelated noah entry should produce an arc whose focus NPC is mara
    and whose theme is drawn from the densest cluster."""
    restore = _isolate_state_file("arc_propose")
    try:
        engine = _make_stub_engine()
        director = StoryDirector(engine)
        _inject_fake_ledger_entries(director.ledger, [
            {"text": "Mara is hiding contraband under the floorboards.",
             "embedding": _base_embedding(),
             "npc_id": "bess", "kind": "fact", "tick": 1},
            {"text": "Mara was seen near Kael's forge at night.",
             "embedding": _fake_embedding(0.95),
             "npc_id": "bess", "kind": "event", "tick": 2},
            {"text": "Mara refused to answer questions about her shipments.",
             "embedding": _fake_embedding(0.92),
             "npc_id": "kael", "kind": "fact", "tick": 3},
            # Unrelated noah entry — should NOT be pulled into the cluster
            {"text": "Noah is reading old king's letters in his study.",
             "embedding": [0.0, 1.0],
             "npc_id": "noah", "kind": "event", "tick": 4},
        ])
        arc = director.arc_planner.maybe_propose(
            director.ledger, ["kael", "bess", "noah"], current_tick=5,
        )
        assert arc is not None, "expected a proposal from the clustered ledger"
        assert arc.status == "active"
        assert "Mara" in arc.theme or "mara" in arc.theme.lower(), arc.theme
        # The cluster is ledger-entry-labeled with bess+kael; noah should be
        # excluded because its entry is outside the cluster
        assert "noah" not in arc.focus_npcs, arc.focus_npcs
        assert set(arc.focus_npcs).issubset({"bess", "kael"}), arc.focus_npcs
        assert arc in director.arc_planner.active_arcs()
    finally:
        restore()
    print("  [PASS] arc_planner_proposes_from_clustered_ledger")


def test_arc_planner_skips_with_too_few_entries():
    """Below the minimum-ledger-entries threshold, maybe_propose must
    return None and not create an arc."""
    restore = _isolate_state_file("arc_thin")
    try:
        engine = _make_stub_engine()
        director = StoryDirector(engine)
        _inject_fake_ledger_entries(director.ledger, [
            {"text": "one", "embedding": _base_embedding(),
             "npc_id": "mara", "kind": "fact", "tick": 1},
            {"text": "two", "embedding": _fake_embedding(0.95),
             "npc_id": "mara", "kind": "event", "tick": 2},
        ])
        arc = director.arc_planner.maybe_propose(
            director.ledger, ["mara", "kael"], current_tick=3,
        )
        assert arc is None
        assert director.arc_planner.active_arcs() == []
        assert director.arc_planner.arcs == []
    finally:
        restore()
    print("  [PASS] arc_planner_skips_with_too_few_entries")


def test_arc_planner_caps_at_max_concurrent_arcs():
    """With _MAX_CONCURRENT_ARCS active arcs, maybe_propose must not
    create a new one. Once the cap is reached, the planner holds
    steady until one of the active arcs resolves."""
    import npc_engine.story_director as sd_mod
    restore = _isolate_state_file("arc_cap")
    try:
        engine = _make_stub_engine()
        director = StoryDirector(engine)
        # Fill the planner to the concurrent cap
        for i in range(sd_mod._MAX_CONCURRENT_ARCS):
            arc = NarrativeArc(
                id=f"arc_{i}", theme=f"theme {i}", focus_npcs=[f"npc_{i}"],
                beat_goals=["seed — x", "escalate — x", "confront — x", "resolve — x"],
                current_beat=0, status="active",
                started_at_tick=0, last_advanced_at_tick=0,
            )
            director.arc_planner.arcs.append(arc)
            director.arc_planner.active_arc_ids.append(arc.id)
        _inject_fake_ledger_entries(director.ledger, [
            {"text": f"entry {i}", "embedding": _base_embedding(),
             "npc_id": "mara", "kind": "fact", "tick": i}
            for i in range(6)
        ])
        arc = director.arc_planner.maybe_propose(
            director.ledger, ["mara", "kael"], current_tick=10,
        )
        assert arc is None, "proposal must be blocked at cap"
        assert len(director.arc_planner.active_arcs()) == sd_mod._MAX_CONCURRENT_ARCS
    finally:
        restore()
    print("  [PASS] arc_planner_caps_at_max_concurrent_arcs")


def test_arc_planner_proposal_excludes_npcs_in_active_casts():
    """maybe_propose must not form a new arc on NPCs already in an
    active arc's cast. Otherwise the densest cluster would keep
    being the saturated thread and proposals would duplicate."""
    restore = _isolate_state_file("arc_exclude_used")
    try:
        engine = _make_stub_engine()
        director = StoryDirector(engine)
        # Active arc already owns bess
        existing = NarrativeArc(
            id="arc_bess", theme="bess thread", focus_npcs=["bess"],
            beat_goals=["seed — x", "escalate — x", "confront — x", "resolve — x"],
            current_beat=0, status="active",
            started_at_tick=0, last_advanced_at_tick=0,
        )
        director.arc_planner.arcs.append(existing)
        director.arc_planner.active_arc_ids.append(existing.id)

        # Seed the ledger with a bess-dominated cluster (which would
        # otherwise be the densest) plus a kael cluster of at least
        # _ARC_PROPOSAL_MIN_LEDGER_ENTRIES entries so the filtered
        # recent list can still clear the proposal threshold.
        _inject_fake_ledger_entries(director.ledger, [
            # 3 bess entries — dense but excluded
            {"text": "bess a", "embedding": _base_embedding(),
             "npc_id": "bess", "kind": "fact", "tick": 1},
            {"text": "bess b", "embedding": _fake_embedding(0.95),
             "npc_id": "bess", "kind": "fact", "tick": 2},
            {"text": "bess c", "embedding": _fake_embedding(0.92),
             "npc_id": "bess", "kind": "event", "tick": 3},
            # 4 kael entries — clustered around a different angle
            # from bess so the NPC filter leaves a dense cluster
            {"text": "kael hammers stolen", "embedding": [0.2, 0.98],
             "npc_id": "kael", "kind": "event", "tick": 4},
            {"text": "kael suspects mara", "embedding": [0.25, 0.97],
             "npc_id": "kael", "kind": "fact", "tick": 5},
            {"text": "kael investigates", "embedding": [0.22, 0.975],
             "npc_id": "kael", "kind": "event", "tick": 6},
            {"text": "kael finds clue", "embedding": [0.18, 0.984],
             "npc_id": "kael", "kind": "fact", "tick": 7},
        ])
        arc = director.arc_planner.maybe_propose(
            director.ledger, ["bess", "kael"], current_tick=10,
        )
        # The new proposal must not be about bess (excluded) — it
        # should form around kael instead
        assert arc is not None, "should still propose, excluding bess"
        assert "bess" not in arc.focus_npcs, arc.focus_npcs
        assert "kael" in arc.focus_npcs, arc.focus_npcs
        # Both arcs should now be active
        assert len(director.arc_planner.active_arcs()) == 2
    finally:
        restore()
    print("  [PASS] arc_planner_proposal_excludes_npcs_in_active_casts")


def test_arc_for_focus_returns_matching_arc():
    """arc_for_focus(npc) returns the one active arc whose cast
    contains ``npc`` — or None if none do."""
    restore = _isolate_state_file("arc_for_focus")
    try:
        engine = _make_stub_engine()
        director = StoryDirector(engine)
        arc_a = NarrativeArc(
            id="arc_a", theme="a", focus_npcs=["bess", "mara"],
            beat_goals=["a", "b", "c", "d"],
            current_beat=0, status="active",
            started_at_tick=0, last_advanced_at_tick=0,
            touches_since_last_advance=3,
        )
        arc_b = NarrativeArc(
            id="arc_b", theme="b", focus_npcs=["kael"],
            beat_goals=["a", "b", "c", "d"],
            current_beat=0, status="active",
            started_at_tick=0, last_advanced_at_tick=0,
            touches_since_last_advance=0,
        )
        director.arc_planner.arcs.extend([arc_a, arc_b])
        director.arc_planner.active_arc_ids.extend(["arc_a", "arc_b"])

        # bess is in arc_a only
        assert director.arc_planner.arc_for_focus("bess") is arc_a
        # kael is in arc_b only
        assert director.arc_planner.arc_for_focus("kael") is arc_b
        # noah is in neither
        assert director.arc_planner.arc_for_focus("noah") is None
        # None / empty
        assert director.arc_planner.arc_for_focus(None) is None
        assert director.arc_planner.arc_for_focus("") is None
    finally:
        restore()
    print("  [PASS] arc_for_focus_returns_matching_arc")


def test_arc_for_focus_prefers_weakest_thread_on_overlap():
    """If an NPC is in multiple active arc casts (defensive path —
    proposal normally excludes overlap), prefer the arc with the
    fewest touches since its last advance. That lets starved arcs
    catch up."""
    restore = _isolate_state_file("arc_weakest")
    try:
        engine = _make_stub_engine()
        director = StoryDirector(engine)
        strong = NarrativeArc(
            id="arc_strong", theme="strong", focus_npcs=["bess"],
            beat_goals=["a", "b", "c", "d"],
            current_beat=0, status="active",
            started_at_tick=0, last_advanced_at_tick=0,
            touches_since_last_advance=5,  # accumulating fast
        )
        weak = NarrativeArc(
            id="arc_weak", theme="weak", focus_npcs=["bess"],
            beat_goals=["a", "b", "c", "d"],
            current_beat=0, status="active",
            started_at_tick=0, last_advanced_at_tick=0,
            touches_since_last_advance=0,  # starved
        )
        director.arc_planner.arcs.extend([strong, weak])
        director.arc_planner.active_arc_ids.extend(["arc_strong", "arc_weak"])
        assert director.arc_planner.arc_for_focus("bess") is weak
    finally:
        restore()
    print("  [PASS] arc_for_focus_prefers_weakest_thread_on_overlap")


def test_record_cast_touch_bumps_all_matching_arcs():
    """record_cast_touch iterates ALL active arcs whose cast contains
    the NPC, not just one. This is defensive — proposal normally
    excludes overlap — but if a future change introduces overlap we
    want both arcs to accumulate touches correctly."""
    restore = _isolate_state_file("arc_bump_all")
    try:
        engine = _make_stub_engine()
        director = StoryDirector(engine)
        arc_a = NarrativeArc(
            id="arc_a", theme="a", focus_npcs=["bess", "mara"],
            beat_goals=["a", "b", "c", "d"],
            current_beat=0, status="active",
            started_at_tick=0, last_advanced_at_tick=0,
        )
        arc_b = NarrativeArc(
            id="arc_b", theme="b", focus_npcs=["bess"],
            beat_goals=["a", "b", "c", "d"],
            current_beat=0, status="active",
            started_at_tick=0, last_advanced_at_tick=0,
        )
        director.arc_planner.arcs.extend([arc_a, arc_b])
        director.arc_planner.active_arc_ids.extend(["arc_a", "arc_b"])

        director.arc_planner.record_cast_touch("bess")
        assert arc_a.touches_since_last_advance == 1
        assert arc_b.touches_since_last_advance == 1

        # Touching mara bumps only arc_a
        director.arc_planner.record_cast_touch("mara")
        assert arc_a.touches_since_last_advance == 2
        assert arc_b.touches_since_last_advance == 1
    finally:
        restore()
    print("  [PASS] record_cast_touch_bumps_all_matching_arcs")


def test_advance_if_beat_met_advances_multiple_arcs():
    """When several active arcs have met the threshold, a single
    advance_if_beat_met call advances all of them."""
    import npc_engine.story_director as sd_mod
    restore = _isolate_state_file("arc_advance_many")
    try:
        engine = _make_stub_engine()
        director = StoryDirector(engine)
        threshold = sd_mod._ARC_BEAT_ADVANCE_THRESHOLD
        arc_a = NarrativeArc(
            id="arc_a", theme="a", focus_npcs=["bess"],
            beat_goals=["a", "b", "c", "d"],
            current_beat=0, status="active",
            started_at_tick=0, last_advanced_at_tick=0,
            touches_since_last_advance=threshold,
        )
        arc_b = NarrativeArc(
            id="arc_b", theme="b", focus_npcs=["kael"],
            beat_goals=["a", "b", "c", "d"],
            current_beat=0, status="active",
            started_at_tick=0, last_advanced_at_tick=0,
            touches_since_last_advance=threshold,
        )
        arc_c_below = NarrativeArc(
            id="arc_c", theme="c", focus_npcs=["noah"],
            beat_goals=["a", "b", "c", "d"],
            current_beat=0, status="active",
            started_at_tick=0, last_advanced_at_tick=0,
            touches_since_last_advance=threshold - 1,
        )
        director.arc_planner.arcs.extend([arc_a, arc_b, arc_c_below])
        director.arc_planner.active_arc_ids.extend(["arc_a", "arc_b", "arc_c"])

        advanced = director.arc_planner.advance_if_beat_met(current_tick=5)
        assert advanced == 2
        assert arc_a.current_beat == 1
        assert arc_b.current_beat == 1
        assert arc_c_below.current_beat == 0  # below threshold
        assert arc_a.touches_since_last_advance == 0
        assert arc_b.touches_since_last_advance == 0
    finally:
        restore()
    print("  [PASS] advance_if_beat_met_advances_multiple_arcs")


def test_arc_planner_advances_beat_after_n_touches():
    """When the active arc's touches_since_last_advance reaches the
    threshold, the current beat should bump by one. Below the
    threshold, no advance. Counter resets to 0 on advance."""
    import npc_engine.story_director as sd_mod
    restore = _isolate_state_file("arc_advance")
    try:
        engine = _make_stub_engine()
        director = StoryDirector(engine)
        arc = NarrativeArc(
            id="arc_advance_test", theme="mara smuggling",
            focus_npcs=["mara"],
            beat_goals=["seed — x", "escalate — x", "confront — x", "resolve — x"],
            current_beat=0, status="active",
            started_at_tick=1, last_advanced_at_tick=1,
        )
        director.arc_planner.arcs.append(arc)
        director.arc_planner.active_arc_ids.append(arc.id)

        threshold = sd_mod._ARC_BEAT_ADVANCE_THRESHOLD

        # Below threshold — should NOT advance
        arc.touches_since_last_advance = threshold - 1
        advanced = director.arc_planner.advance_if_beat_met(current_tick=5)
        assert advanced == 0, "must not advance below threshold"
        assert arc.current_beat == 0, arc.current_beat
        assert arc.touches_since_last_advance == threshold - 1, (
            "below-threshold call should not reset the counter"
        )

        # At threshold — SHOULD advance and counter resets
        arc.touches_since_last_advance = threshold
        advanced = director.arc_planner.advance_if_beat_met(current_tick=6)
        assert advanced == 1
        assert arc.current_beat == 1, arc.current_beat
        assert arc.status == "active"
        assert arc.touches_since_last_advance == 0, "counter must reset on advance"
        assert arc.last_advanced_at_tick == 6
    finally:
        restore()
    print("  [PASS] arc_planner_advances_beat_after_n_touches")


def test_arc_planner_record_cast_touch_bumps_counter():
    """record_cast_touch should bump the active arc's counter ONLY
    when the NPC is in the cast."""
    restore = _isolate_state_file("arc_record_touch")
    try:
        engine = _make_stub_engine()
        director = StoryDirector(engine)
        arc = NarrativeArc(
            id="arc_touch_test", theme="x",
            focus_npcs=["mara", "kael"],
            beat_goals=["a", "b", "c", "d"],
            current_beat=0, status="active",
            started_at_tick=1, last_advanced_at_tick=1,
        )
        director.arc_planner.arcs.append(arc)
        director.arc_planner.active_arc_ids.append(arc.id)

        # Touch cast NPCs — counter bumps
        director.arc_planner.record_cast_touch("mara")
        director.arc_planner.record_cast_touch("kael")
        assert arc.touches_since_last_advance == 2

        # Touch non-cast NPC — counter does NOT bump
        director.arc_planner.record_cast_touch("bess")
        assert arc.touches_since_last_advance == 2

        # None / empty string — no-op
        director.arc_planner.record_cast_touch(None)
        director.arc_planner.record_cast_touch("")
        assert arc.touches_since_last_advance == 2
    finally:
        restore()
    print("  [PASS] arc_planner_record_cast_touch_bumps_counter")


def test_arc_planner_record_cast_touch_ignores_inactive_arcs():
    """record_cast_touch is a no-op when there's no active arc or
    the arc is resolved."""
    restore = _isolate_state_file("arc_touch_inactive")
    try:
        engine = _make_stub_engine()
        director = StoryDirector(engine)
        # No active arc
        director.arc_planner.record_cast_touch("mara")
        assert director.arc_planner.active_arcs() == []

        # Add a resolved arc
        resolved = NarrativeArc(
            id="arc_done", theme="x", focus_npcs=["mara"],
            beat_goals=["a", "b", "c", "d"],
            current_beat=4, status="resolved",
            started_at_tick=1, last_advanced_at_tick=10,
        )
        director.arc_planner.arcs.append(resolved)
        director.arc_planner.active_arc_ids = []
        director.arc_planner.record_cast_touch("mara")
        # Resolved arc's counter stays at 0
        assert resolved.touches_since_last_advance == 0
    finally:
        restore()
    print("  [PASS] arc_planner_record_cast_touch_ignores_inactive_arcs")


def test_arc_planner_resolves_after_all_beats():
    """When the final beat advances, the arc should flip to resolved
    and its id should be removed from active_arc_ids."""
    import npc_engine.story_director as sd_mod
    restore = _isolate_state_file("arc_resolve")
    try:
        engine = _make_stub_engine()
        director = StoryDirector(engine)
        arc = NarrativeArc(
            id="arc_resolve_test", theme="mara smuggling",
            focus_npcs=["mara"],
            beat_goals=["seed — x", "escalate — x", "confront — x", "resolve — x"],
            current_beat=3, status="active",
            started_at_tick=1, last_advanced_at_tick=6,
        )
        director.arc_planner.arcs.append(arc)
        director.arc_planner.active_arc_ids.append(arc.id)

        arc.touches_since_last_advance = sd_mod._ARC_BEAT_ADVANCE_THRESHOLD
        advanced = director.arc_planner.advance_if_beat_met(current_tick=10)
        assert advanced == 1
        assert arc.current_beat == 4
        assert arc.is_complete
        assert arc.status == "resolved"
        assert director.arc_planner.active_arcs() == []
        assert arc.id not in director.arc_planner.active_arc_ids
    finally:
        restore()
    print("  [PASS] arc_planner_resolves_after_all_beats")


def test_prompt_contains_active_arc_block():
    """When an arc is active, _build_prompt must include an ACTIVE
    NARRATIVE ARC section with the theme and current beat goal."""
    restore = _isolate_state_file("arc_prompt")
    try:
        engine = _make_stub_engine()
        director = StoryDirector(engine)
        arc = NarrativeArc(
            id="arc_prompt_test", theme="Mara's smuggling ring",
            focus_npcs=["mara", "kael"],
            beat_goals=["seed — introduce suspicion", "escalate — x", "confront — x", "resolve — x"],
            current_beat=0, status="active",
            started_at_tick=1, last_advanced_at_tick=1,
        )
        director.arc_planner.arcs.append(arc)
        director.arc_planner.active_arc_ids.append(arc.id)

        prompt = director._build_prompt("world snap", focus_npc="kael", action_kind="event")
        assert "ACTIVE NARRATIVE ARC" in prompt, prompt
        assert "Mara's smuggling ring" in prompt, prompt
        assert "introduce suspicion" in prompt, prompt
        # The forced focus block must STILL be the last guidance (recency
        # bias belongs to focus/kind, not the arc).
        arc_idx = prompt.index("ACTIVE NARRATIVE ARC")
        focus_idx = prompt.index("FOCUS NPC")
        assert arc_idx < focus_idx, "arc block must come before FOCUS NPC block"
    finally:
        restore()
    print("  [PASS] prompt_contains_active_arc_block")


def test_arc_planner_persists_and_reloads():
    """Saving and reloading should preserve arcs, active_arc_ids, and
    the cooldown tick counter."""
    restore = _isolate_state_file("arc_persist")
    try:
        import npc_engine.story_director as sd_mod
        planner1 = ArcPlanner(sd_mod.ARCS_FILE)
        planner1.arcs.append(NarrativeArc(
            id="arc_p1", theme="theme one",
            focus_npcs=["mara", "kael"],
            beat_goals=["a", "b", "c", "d"],
            current_beat=2, status="active",
            started_at_tick=3, last_advanced_at_tick=7,
        ))
        planner1.active_arc_ids = ["arc_p1"]
        planner1._last_proposal_attempt_tick = 9
        planner1.save()

        planner2 = ArcPlanner(sd_mod.ARCS_FILE)
        assert len(planner2.arcs) == 1
        restored = planner2.arcs[0]
        assert restored.id == "arc_p1"
        assert restored.focus_npcs == ["mara", "kael"]
        assert restored.current_beat == 2
        assert restored.last_advanced_at_tick == 7
        assert planner2.active_arc_ids == ["arc_p1"]
        assert planner2._last_proposal_attempt_tick == 9
        assert restored in planner2.active_arcs()
    finally:
        restore()
    print("  [PASS] arc_planner_persists_and_reloads")


def test_arc_planner_cooldown_prevents_immediate_reproposal():
    """After a proposal (or attempted proposal), the planner must sit
    out for _ARC_PROPOSAL_COOLDOWN_TICKS before trying again — even if
    the ledger grows substantially."""
    restore = _isolate_state_file("arc_cooldown")
    try:
        import npc_engine.story_director as sd_mod
        engine = _make_stub_engine()
        director = StoryDirector(engine)
        _inject_fake_ledger_entries(director.ledger, [
            {"text": "entry a", "embedding": _base_embedding(),
             "npc_id": "mara", "kind": "fact", "tick": 1},
            {"text": "entry b", "embedding": _fake_embedding(0.95),
             "npc_id": "mara", "kind": "event", "tick": 2},
            {"text": "entry c", "embedding": _fake_embedding(0.9),
             "npc_id": "kael", "kind": "fact", "tick": 3},
            {"text": "entry d", "embedding": _fake_embedding(0.92),
             "npc_id": "kael", "kind": "event", "tick": 4},
        ])
        arc1 = director.arc_planner.maybe_propose(
            director.ledger, ["mara", "kael"], current_tick=5,
        )
        assert arc1 is not None
        # Resolve the first arc so the concurrent-cap guard doesn't
        # block the re-propose attempts below. Use the full resolve
        # path: flip status and drop the id from active_arc_ids.
        arc1.status = "resolved"
        director.arc_planner.active_arc_ids = []

        # Immediately try again at tick 6 — cooldown has NOT elapsed
        arc2 = director.arc_planner.maybe_propose(
            director.ledger, ["mara", "kael"], current_tick=6,
        )
        assert arc2 is None, "cooldown must block immediate re-proposal"

        # After cooldown elapses, a new proposal should succeed
        arc3 = director.arc_planner.maybe_propose(
            director.ledger,
            ["mara", "kael"],
            current_tick=5 + sd_mod._ARC_PROPOSAL_COOLDOWN_TICKS,
        )
        assert arc3 is not None
        assert arc3.id != arc1.id
    finally:
        restore()
    print("  [PASS] arc_planner_cooldown_prevents_immediate_reproposal")


# ── NPC bio injection tests ─────────────────────────────────────

def _attach_goals(engine, npc_id: str, goals: list[dict]) -> None:
    """Attach a stub GoalsCapability to an NPC so _peek_npc_goals can
    find it. Mirrors the shape real NPC capability managers expose."""
    mgr = engine.pie.capability_managers.get(npc_id)
    if mgr is None:
        mgr = _StubCapabilityManager()
        engine.pie.capability_managers[npc_id] = mgr
    mgr.capabilities["goals"] = _StubGoalsCap(goals)


def test_peek_npc_goals_reads_priority_sorted():
    """_peek_npc_goals should return the capability's priority-sorted
    goal list, and an empty list when the NPC has no goals."""
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    _attach_goals(engine, "mara", [
        {"id": "expand_trade", "description": "Grow the guild", "priority": 7},
        {"id": "hide_counterfeits", "description": "Hide the fake steel", "priority": 9},
    ])
    goals = director._peek_npc_goals("mara")
    assert len(goals) == 2
    assert goals[0]["id"] == "hide_counterfeits", goals  # priority 9 first
    assert goals[1]["id"] == "expand_trade", goals
    # NPC without a goals cap
    assert director._peek_npc_goals("kael") == []
    # Unknown NPC
    assert director._peek_npc_goals("nobody") == []
    print("  [PASS] peek_npc_goals_reads_priority_sorted")


def test_focus_npc_bio_contains_personality_goals_and_knowledge():
    """_build_focus_npc_bio should pull personality, goals, and
    personal_knowledge into a single multi-line block."""
    engine = _make_stub_engine()
    mara = engine.pie.npc_knowledge.get("kael")  # use existing stub slot
    mara.identity = {
        "name": "Kael", "role": "blacksmith",
        "personality": "Gruff, proud, quick to anger but slow to forgive.",
    }
    mara.personal_knowledge = [
        "I suspect Mara is selling counterfeit steel.",
        "I've been losing hammers — someone is stealing them.",
    ]
    mara.world_facts = ["Ashenvale has a blacksmith (me) and a merchant (Mara)."]
    _attach_goals(engine, "kael", [
        {"id": "find_thief", "description": "Catch whoever is stealing my hammers", "priority": 8},
        {"id": "expose_mara", "description": "Prove Mara is selling fake steel", "priority": 9},
    ])
    director = StoryDirector(engine)
    bio = director._build_focus_npc_bio("kael")
    assert bio is not None
    assert "FOCUS NPC BIO: kael" in bio
    assert "Gruff, proud" in bio
    assert "Driving goals:" in bio
    # Priority-9 should appear before priority-8 in the bio
    idx_p9 = bio.index("[p9]")
    idx_p8 = bio.index("[p8]")
    assert idx_p9 < idx_p8, bio
    assert "Prove Mara is selling fake steel" in bio, bio
    assert "Catch whoever is stealing my hammers" in bio, bio
    assert "Private knowledge" in bio
    assert "counterfeit steel" in bio
    assert "Their view of the world" in bio
    print("  [PASS] focus_npc_bio_contains_personality_goals_and_knowledge")


def test_focus_npc_bio_returns_none_when_npc_is_bare():
    """An NPC with only {name, role} (no personality, goals, pk, wf)
    produces no bio — the caller should skip the block entirely."""
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    # Stub NPCs default to empty personal_knowledge/world_facts and no
    # personality in identity — no goals attached either
    bio = director._build_focus_npc_bio("noah")
    assert bio is None, bio
    print("  [PASS] focus_npc_bio_returns_none_when_npc_is_bare")


def test_world_snapshot_roster_includes_top_goal():
    """The per-NPC line in the snapshot should surface 'wants: ...'
    when the NPC has an active goal."""
    engine = _make_stub_engine()
    _attach_goals(engine, "bess", [
        {"id": "protect_village", "description": "Keep Ashenvale safe from the Silverwood threats", "priority": 8},
    ])
    director = StoryDirector(engine)
    snapshot = director._world_snapshot()
    # The bess line should include the wants suffix
    bess_lines = [ln for ln in snapshot.split("\n") if "bess" in ln and "(innkeeper)" in ln]
    assert bess_lines, f"bess line missing from snapshot: {snapshot}"
    assert "wants:" in bess_lines[0], bess_lines[0]
    assert "Silverwood" in bess_lines[0], bess_lines[0]
    # NPCs without goals should not emit a wants suffix
    kael_lines = [ln for ln in snapshot.split("\n") if "kael" in ln and "(blacksmith)" in ln]
    assert kael_lines
    assert "wants:" not in kael_lines[0], kael_lines[0]
    print("  [PASS] world_snapshot_roster_includes_top_goal")


def test_prompt_contains_bio_block_for_focus_npc():
    """When a focus NPC has bio data, _build_prompt should inject a
    FOCUS NPC BIO block between ACTIVE NARRATIVE ARC and FOCUS NPC."""
    engine = _make_stub_engine()
    kael = engine.pie.npc_knowledge.get("kael")
    kael.identity = {
        "name": "Kael", "role": "blacksmith",
        "personality": "Gruff and hardworking.",
    }
    kael.personal_knowledge = ["I suspect someone is stealing my hammers."]
    _attach_goals(engine, "kael", [
        {"id": "catch_thief", "description": "Catch the hammer thief", "priority": 8},
    ])
    director = StoryDirector(engine)
    prompt = director._build_prompt("world snap", focus_npc="kael", action_kind="event")
    assert "FOCUS NPC BIO: kael" in prompt
    assert "Gruff and hardworking" in prompt
    assert "Catch the hammer thief" in prompt
    assert "I suspect someone is stealing my hammers." in prompt
    # Ordering: BIO must come before the FOCUS NPC directive block
    bio_idx = prompt.index("FOCUS NPC BIO")
    focus_idx = prompt.index("FOCUS NPC FOR THIS TICK")
    assert bio_idx < focus_idx, (
        "bio block must come before the forced-focus directive "
        "(directive holds recency, bio provides context)"
    )
    print("  [PASS] prompt_contains_bio_block_for_focus_npc")


def test_prompt_skips_bio_block_when_npc_is_bare():
    """A bio-less NPC should not produce a FOCUS NPC BIO header in
    the prompt — the block is skipped entirely."""
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    prompt = director._build_prompt("world snap", focus_npc="noah", action_kind="event")
    assert "FOCUS NPC BIO" not in prompt
    # The forced-focus directive should still be there
    assert "FOCUS NPC FOR THIS TICK" in prompt
    print("  [PASS] prompt_skips_bio_block_when_npc_is_bare")


# ── Self-repetition precheck tests ──────────────────────────────

def _fake_ledger_check(warning: Optional[dict]):
    """Return a monkey-patch replacement for ``ledger.check`` that
    always yields the same canned warning. None means no match.
    Accepts (and ignores) the ``restrict_to_npc`` kwarg so tests can
    drive the real precheck code path which always passes it."""
    def _check(_text, restrict_to_npc=None):
        return warning
    return _check


def test_precheck_self_repetition_fires_on_high_sim_same_npc_recent():
    """A candidate that's near-duplicate (>=0.75) of a recent
    same-NPC Director entry must fire the precheck."""
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    director.tick_count = 5
    director.ledger.check = _fake_ledger_check({
        "similarity": 0.84,
        "matches_text": "Bess drops a tray of hot soup",
        "matches_npc": "bess",
        "matches_kind": "event",
        "matches_tick": 3,
    })
    action = {"action": "event", "target": "bess",
              "event": "Bess fumbles a bowl of steaming stew"}
    warning = director._precheck_self_repetition(action)
    assert warning is not None
    assert warning["matches_tick"] == 3
    print("  [PASS] precheck_self_repetition_fires_on_high_sim_same_npc_recent")


def test_precheck_self_repetition_restricts_to_same_npc():
    """The precheck must pass restrict_to_npc to the ledger so the
    similarity search only considers same-NPC entries. Cross-NPC
    similarity is gossip propagation, not self-repetition — and
    without the restriction, a high-similarity cross-NPC match would
    mask a lower-similarity same-NPC match sitting below it."""
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    director.tick_count = 5
    # Capture the kwargs so we can verify the NPC filter was passed
    captured = {}
    def spy_check(text, restrict_to_npc=None):
        captured["npc"] = restrict_to_npc
        return None
    director.ledger.check = spy_check
    action = {"action": "event", "target": "bess",
              "event": "Bess does something"}
    director._precheck_self_repetition(action)
    assert captured["npc"] == "bess", captured
    print("  [PASS] precheck_self_repetition_restricts_to_same_npc")


def test_ledger_check_filters_by_restrict_to_npc():
    """FactLedger.check(restrict_to_npc=X) must only compare against
    entries with matching npc_id. Unit-tests the filter directly."""
    import numpy as np
    tmp_path = NPC_ROOT / "data" / "story_director" / "_tmp_ledger_restrict.json"
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        ledger = FactLedger(tmp_path, threshold=0.5)
        ledger._np = np
        ledger._embedder = False
        # Seed with a high-similarity kael entry and a lower-similarity
        # bess entry against the same query embedding [1, 0]
        ledger.entries = [
            {"text": "Kael near-duplicate",
             "embedding": [0.99, 0.14],  # sim=0.99 to [1,0]
             "npc_id": "kael", "kind": "event", "tick": 1},
            {"text": "Bess weaker match",
             "embedding": [0.75, 0.66],  # sim=0.75 to [1,0]
             "npc_id": "bess", "kind": "event", "tick": 2},
        ]
        # Stub the query to return the base embedding directly
        ledger._encode = lambda text: np.array([1.0, 0.0], dtype=float)

        # No restriction: top match should be kael (sim 0.99)
        w_open = ledger.check("anything")
        assert w_open is not None
        assert w_open["matches_npc"] == "kael"

        # Restricted to bess: should return bess (sim 0.75), NOT kael
        w_bess = ledger.check("anything", restrict_to_npc="bess")
        assert w_bess is not None
        assert w_bess["matches_npc"] == "bess"
        assert w_bess["similarity"] < 0.99  # not the kael match

        # Restricted to an unknown NPC: should return None
        w_none = ledger.check("anything", restrict_to_npc="nobody")
        assert w_none is None
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    print("  [PASS] ledger_check_filters_by_restrict_to_npc")


def test_precheck_self_repetition_ignores_low_similarity():
    """Matches below the 0.75 threshold are thematic overlap, not
    self-repetition — they should not fire the precheck."""
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    director.tick_count = 5
    director.ledger.check = _fake_ledger_check({
        "similarity": 0.62,
        "matches_text": "Something else about bess",
        "matches_npc": "bess",
        "matches_kind": "fact",
        "matches_tick": 3,
    })
    action = {"action": "fact", "npc_id": "bess", "fact": "Bess hears a rumor"}
    assert director._precheck_self_repetition(action) is None
    print("  [PASS] precheck_self_repetition_ignores_low_similarity")


def test_precheck_self_repetition_ignores_old_matches():
    """Stale matches from beyond the lookback window should be
    allowed through — plot threads can legitimately echo old beats
    after enough ticks pass. The window is set by
    _SELF_REPETITION_LOOKBACK_TICKS so the test derives its ticks
    from the module constant rather than hardcoding."""
    import npc_engine.story_director as sd_mod
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    window = sd_mod._SELF_REPETITION_LOOKBACK_TICKS
    # Current tick_count, candidate at tick_count+1. Put the match
    # JUST beyond the window (age = window + 2).
    director.tick_count = window + 5
    old_tick = director.tick_count + 1 - (window + 2)
    director.ledger.check = _fake_ledger_check({
        "similarity": 0.90,
        "matches_text": "Bess ancient event",
        "matches_npc": "bess",
        "matches_kind": "event",
        "matches_tick": old_tick,
    })
    action = {"action": "event", "target": "bess", "event": "Bess does something"}
    assert director._precheck_self_repetition(action) is None, (
        "match beyond lookback window should not fire"
    )
    # Same setup but match WITHIN the window (age = window - 1).
    recent_tick = director.tick_count + 1 - (window - 1)
    director.ledger.check = _fake_ledger_check({
        "similarity": 0.90,
        "matches_text": "Bess recent event",
        "matches_npc": "bess",
        "matches_kind": "event",
        "matches_tick": recent_tick,
    })
    assert director._precheck_self_repetition(action) is not None
    print("  [PASS] precheck_self_repetition_ignores_old_matches")


def test_precheck_self_repetition_ignores_all_target():
    """target='all' is a global event without a specific NPC — can't
    self-repeat a global, so the precheck should skip."""
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    director.tick_count = 5
    director.ledger.check = _fake_ledger_check({
        "similarity": 0.90,
        "matches_text": "Village-wide event",
        "matches_npc": "all",
        "matches_kind": "event",
        "matches_tick": 4,
    })
    action = {"action": "event", "target": "all", "event": "Something"}
    assert director._precheck_self_repetition(action) is None
    print("  [PASS] precheck_self_repetition_ignores_all_target")


def test_self_repetition_retry_dispatches_second_response():
    """Full integration: when the ledger precheck fires self-repetition,
    the worker retries once and the SECOND response is what gets
    dispatched. Flag retried_after_self_repetition on the result."""
    restore = _isolate_state_file("selfrep_retry")
    try:
        engine = _make_stub_engine(responses=[
            # First response: will be flagged as a near-duplicate
            '{"action": "event", "target": "bess", '
            '"event": "Bess drops a tray of hot soup again."}',
            # Retry response: novel content
            '{"action": "event", "target": "bess", '
            '"event": "Bess slips out the back door to meet a mysterious courier."}',
        ])
        director = StoryDirector(engine)
        # Pin rotation so this tick targets bess with an event
        director.recent_player_actions = [{
            "at": "2030-01-01T00:00:00+00:00",
            "tick_at_time": 0,
            "text": "Player visited Bess.",
            "target": "bess",
        }]
        director._kind_rotation_index = _ACTION_KIND_ROTATION_INDEX_FOR("event")
        # Monkey-patch ledger.check so the first attempt fires
        # self-repetition. Note both _precheck_contradiction and
        # _precheck_self_repetition call ledger.check on the first
        # attempt, so we need the warning to persist for calls 1 AND
        # 2. Calls 3+ (from the retry path or the dispatch layer)
        # get None so the retry dispatches cleanly.
        selfrep_warning = {
            "similarity": 0.88,
            "matches_text": "Bess drops a hot soup tray",
            "matches_npc": "bess",
            "matches_kind": "event",
            "matches_tick": 1,
        }
        call_count = {"n": 0}
        def staged_check(_text, restrict_to_npc=None):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                return selfrep_warning
            return None
        director.ledger.check = staged_check

        result = director.tick()
        assert len(engine.pie.base_model.prompts) == 2, (
            f"expected two LLM calls (first + retry), got "
            f"{len(engine.pie.base_model.prompts)}"
        )
        assert "mysterious courier" in str(result["action"]), result
        assert result["dispatch"].get("retried_after_self_repetition") is True, (
            result["dispatch"]
        )
    finally:
        restore()
    print("  [PASS] self_repetition_retry_dispatches_second_response")


def test_self_repetition_retry_falls_back_to_original_on_noop():
    """On 3B the retry nudge sometimes degenerates to a noop, which
    would drop the content entirely. Self-rep retries must fall back
    to the original action when the retry returns a noop — slight
    repetition is better than silence."""
    restore = _isolate_state_file("selfrep_noop")
    try:
        engine = _make_stub_engine(responses=[
            # First response: valid content that will be flagged
            '{"action": "event", "target": "bess", '
            '"event": "Bess drops a tray of hot soup again."}',
            # Retry returns a noop — the fallback should kick in
            '{"action": "noop", "reason": "stuck"}',
        ])
        director = StoryDirector(engine)
        director.recent_player_actions = [{
            "at": "2030-01-01T00:00:00+00:00",
            "tick_at_time": 0,
            "text": "Player visited Bess.",
            "target": "bess",
        }]
        director._kind_rotation_index = _ACTION_KIND_ROTATION_INDEX_FOR("event")
        selfrep_warning = {
            "similarity": 0.88,
            "matches_text": "Bess drops a hot soup tray",
            "matches_npc": "bess",
            "matches_kind": "event",
            "matches_tick": 1,
        }
        director.ledger.check = _fake_ledger_check(selfrep_warning)

        result = director.tick()
        # Retry was triggered (two LLM calls happened)
        assert len(engine.pie.base_model.prompts) == 2, engine.pie.base_model.prompts
        # But the DISPATCHED action is the ORIGINAL, not the noop
        assert result["action"].get("action") == "event", result["action"]
        assert "hot soup" in str(result["action"]).lower(), result["action"]
        # Still flagged as retried (for auditing)
        assert result["dispatch"].get("retried_after_self_repetition") is True
    finally:
        restore()
    print("  [PASS] self_repetition_retry_falls_back_to_original_on_noop")


def test_contradiction_takes_precedence_over_self_repetition():
    """If both checks would fire, the contradiction path wins (it's
    the more serious case). The self-repetition check only runs when
    the contradiction check returned None."""
    restore = _isolate_state_file("contra_precedence")
    try:
        engine = _make_stub_engine(responses=[
            '{"action": "fact", "npc_id": "bess", '
            '"fact": "Bess denies the merchant guild rumors."}',
            '{"action": "fact", "npc_id": "bess", '
            '"fact": "Bess keeps her ear to the ground."}',
        ])
        director = StoryDirector(engine)
        director.recent_player_actions = [{
            "at": "2030-01-01T00:00:00+00:00",
            "tick_at_time": 0,
            "text": "Player asked Bess about rumors.",
            "target": "bess",
        }]
        director._kind_rotation_index = _ACTION_KIND_ROTATION_INDEX_FOR("fact")
        # Both contradiction AND high-sim — contradiction should win
        contradiction_warning = {
            "similarity": 0.88,
            "matches_text": "Bess confirmed the guild rumors",
            "matches_npc": "bess",
            "matches_kind": "fact",
            "matches_tick": 1,
            "contradiction": True,
            "nli": {"label": "contradiction", "confidence": 0.95,
                    "is_contradiction": True, "scores": {}},
        }
        call_count = {"n": 0}
        def staged_check(_text, restrict_to_npc=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return contradiction_warning
            return None
        director.ledger.check = staged_check

        result = director.tick()
        # Should be flagged with contradiction, not self_repetition
        assert result["dispatch"].get("retried_after_contradiction") is True
        assert result["dispatch"].get("retried_after_self_repetition") is not True
    finally:
        restore()
    print("  [PASS] contradiction_takes_precedence_over_self_repetition")


# ── Narration mode tests ────────────────────────────────────────

def test_narration_mode_defaults_to_prose():
    """New directors default to prose for backward compatibility —
    existing callers that never set narration_mode get the cinematic
    output style."""
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    assert director.narration_mode == "prose"
    prompt = director._build_prompt("snap", focus_npc="kael", action_kind="event")
    assert "OUTPUT STYLE" not in prompt
    print("  [PASS] narration_mode_defaults_to_prose")


def test_set_narration_mode_reloads_examples_terse_library():
    """Flipping narration_mode to 'terse' via set_narration_mode must
    reload examples from examples_terse.yaml if it exists. Reverse
    flip back to 'prose' should reload the original library."""
    import npc_engine.story_director as sd_mod
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    # Pre-check: prose mode loaded the prose examples (17 entries
    # after the expand commit)
    assert director.narration_mode == "prose"
    prose_count = len(director._examples)
    assert prose_count > 0

    # Terse file must exist alongside the prose file for this test
    assert sd_mod.EXAMPLES_TERSE_FILE.exists(), (
        "examples_terse.yaml missing — the terse library wasn't installed"
    )

    director.set_narration_mode("terse")
    assert director.narration_mode == "terse"
    terse_count = len(director._examples)
    assert terse_count > 0
    # Sanity: the terse examples should be meaningfully shorter than
    # the prose ones. Compare mean word count of the event/fact
    # description fields across loaded examples.
    def mean_wc(items):
        counts = []
        for e in items:
            a = e.get("action", {})
            text = (a.get("event") or a.get("fact") or
                    (a.get("quest", {}) or {}).get("description") or "")
            if text:
                counts.append(len(text.split()))
        return sum(counts) / len(counts) if counts else 0

    # Flip back and compare
    terse_mean = mean_wc(director._examples)
    director.set_narration_mode("prose")
    prose_mean = mean_wc(director._examples)
    assert terse_mean < prose_mean, (
        f"terse examples should be shorter on average: "
        f"terse_mean={terse_mean:.1f} vs prose_mean={prose_mean:.1f}"
    )
    print("  [PASS] set_narration_mode_reloads_examples_terse_library")


def test_set_narration_mode_rejects_unknown_mode():
    """set_narration_mode must reject unknown values to catch typos."""
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    try:
        director.set_narration_mode("cinematic")
    except ValueError:
        print("  [PASS] set_narration_mode_rejects_unknown_mode")
        return
    raise AssertionError("expected ValueError on unknown mode")


def test_narration_mode_terse_injects_output_style_block():
    """Setting narration_mode='terse' injects an OUTPUT STYLE block
    that instructs the model to produce short third-person statements."""
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    director.narration_mode = "terse"
    prompt = director._build_prompt("snap", focus_npc="kael", action_kind="event")
    assert "OUTPUT STYLE" in prompt
    assert "under 25 words" in prompt
    assert "NO internal monologue" in prompt
    # The terse block should come before the forced-focus directive
    style_idx = prompt.index("OUTPUT STYLE")
    focus_idx = prompt.index("FOCUS NPC FOR THIS TICK")
    assert style_idx < focus_idx
    print("  [PASS] narration_mode_terse_injects_output_style_block")


# ── Self-rep retry budget tests ─────────────────────────────────

def test_self_rep_budget_resets_each_tick():
    """The per-tick retry counter must start at 0 on every tick()
    call so budgets don't leak between ticks."""
    restore = _isolate_state_file("budget_reset")
    try:
        engine = _make_stub_engine(responses=[
            '{"action": "noop", "reason": "quiet"}',
            '{"action": "noop", "reason": "still quiet"}',
        ])
        director = StoryDirector(engine)
        # Pretend we burned the budget on a prior tick
        director._self_rep_retries_this_tick = 5
        director.tick()
        # After tick() the counter should be 0 (no retries fired here)
        assert director._self_rep_retries_this_tick == 0
    finally:
        restore()
    print("  [PASS] self_rep_budget_resets_each_tick")


def test_self_rep_budget_blocks_second_retry_in_same_tick():
    """In a 2-action tick where both workers would retry for
    self-repetition, only the first one gets through. The second
    worker's retry is skipped and a skipped_self_rep_retry note
    appears in the dispatch result."""
    import npc_engine.story_director as sd_mod
    restore = _isolate_state_file("budget_block")
    try:
        # Use large cast so the small-cast cap doesn't clamp
        # actions_per_tick=2 down to 1.
        engine = _make_stub_engine(
            profile_specs=_LARGE_STUB_PROFILES,
            responses=[
                # Worker 1 first attempt (flagged), then retry (fresh)
                '{"action": "event", "target": "bess", "event": "Bess scene A"}',
                '{"action": "event", "target": "bess", "event": "Bess fresh after retry"}',
                # Worker 2 first attempt (would be flagged too, but budget = 0 now)
                '{"action": "event", "target": "kael", "event": "Kael scene A"}',
            ],
        )
        director = StoryDirector(engine)
        director._kind_rotation_index = _ACTION_KIND_ROTATION_INDEX_FOR("event")
        # Every precheck fires the same canned warning (we use
        # matches_tick = 1 and the freshly-started director has
        # tick_count = 0, so 0+1-1 = 0 ticks old — well inside the
        # lookback window).
        canned = {
            "similarity": 0.88,
            "matches_text": "Earlier scene",
            "matches_npc": "bess",
            "matches_kind": "event",
            "matches_tick": 1,
        }
        # The precheck calls ledger.check twice per worker (once
        # for contradiction, once for self-rep). Use a staged check
        # that returns the canned warning while the precheck code
        # is running on each worker.
        def fake_check(_text, restrict_to_npc=None):
            # matches_npc mirrors the worker's candidate NPC so the
            # precheck fires for both workers
            return {**canned, "matches_npc": restrict_to_npc or "bess"}
        director.ledger.check = fake_check

        result = director.tick(actions_per_tick=2)
        subs = result["sub_actions"]
        assert len(subs) == 2

        # First worker: retried
        assert subs[0]["dispatch"].get("retried_after_self_repetition") is True
        # Second worker: NOT retried (budget exhausted), and has the
        # skipped_self_rep_retry note on its dispatch
        assert subs[1]["dispatch"].get("retried_after_self_repetition") is not True
        assert "skipped_self_rep_retry" in subs[1]["dispatch"], subs[1]["dispatch"]
        assert subs[1]["dispatch"]["skipped_self_rep_retry"]["reason"] == "budget_exhausted"

        # Only 3 LLM calls total: worker1 initial + worker1 retry + worker2 initial
        assert len(engine.pie.base_model.prompts) == 3, engine.pie.base_model.prompts
    finally:
        restore()
    print("  [PASS] self_rep_budget_blocks_second_retry_in_same_tick")


def test_contradiction_retry_not_budget_gated():
    """Contradiction retries should fire regardless of the self-rep
    budget — they're rare and more serious."""
    restore = _isolate_state_file("budget_contra")
    try:
        # Use large cast so the small-cast cap doesn't clamp
        # actions_per_tick=2 down to 1.
        engine = _make_stub_engine(
            profile_specs=_LARGE_STUB_PROFILES,
            responses=[
                # Worker 1: first flagged as self-rep, retry succeeds
                '{"action": "event", "target": "bess", "event": "Bess scene"}',
                '{"action": "event", "target": "bess", "event": "Bess new scene"}',
                # Worker 2: first flagged as CONTRADICTION, retry should fire
                '{"action": "fact", "npc_id": "kael", "fact": "Kael contradiction"}',
                '{"action": "fact", "npc_id": "kael", "fact": "Kael resolved"}',
            ],
        )
        director = StoryDirector(engine)
        director._kind_rotation_index = _ACTION_KIND_ROTATION_INDEX_FOR("event")

        selfrep_warning = {
            "similarity": 0.88,
            "matches_text": "Earlier Bess scene",
            "matches_npc": "bess",
            "matches_kind": "event",
            "matches_tick": 1,
            # no contradiction flag — this is self-rep only
        }
        contra_warning = {
            "similarity": 0.70,
            "matches_text": "Earlier Kael fact",
            "matches_npc": "kael",
            "matches_kind": "fact",
            "matches_tick": 1,
            "contradiction": True,
            "nli": {"label": "contradiction", "confidence": 0.92,
                    "is_contradiction": True, "scores": {}},
        }
        call_count = {"n": 0}
        def staged_check(_text, restrict_to_npc=None):
            call_count["n"] += 1
            # Worker 1 gets selfrep_warning on both of its 2 calls
            # (contradiction precheck + self-rep precheck). Worker 2
            # gets contra_warning on its 1 call (contradiction precheck
            # fires and stops the chain).
            if call_count["n"] <= 2:
                return selfrep_warning
            return contra_warning
        director.ledger.check = staged_check

        result = director.tick(actions_per_tick=2)
        subs = result["sub_actions"]
        assert len(subs) == 2
        # Worker 1: used the self-rep budget
        assert subs[0]["dispatch"].get("retried_after_self_repetition") is True
        # Worker 2: contradiction retry STILL fires despite self-rep
        # budget being exhausted
        assert subs[1]["dispatch"].get("retried_after_contradiction") is True
        assert subs[1]["dispatch"].get("retried_after_self_repetition") is not True
        # Four LLM calls total
        assert len(engine.pie.base_model.prompts) == 4, engine.pie.base_model.prompts
    finally:
        restore()
    print("  [PASS] contradiction_retry_not_budget_gated")


# ── Intra-bio rotation tests ────────────────────────────────────

def test_is_bio_mentioned_detects_word_overlap():
    """The mention heuristic should fire when a majority of content
    words from a bio item appear in the output text (paraphrase or
    verbatim), and stay silent on unrelated content."""
    engine = _make_stub_engine()
    director = StoryDirector(engine)

    bio = "Overheard the merchant guild planning to raise taxes next season"
    # Near-verbatim quote — clearly a mention
    out1 = "I heard the merchant guild is planning to raise taxes next month."
    assert director._is_bio_mentioned(bio, out1.lower()) is True
    # Paraphrased but same subject — should still hit
    out2 = "bess mentioned the merchant guild was planning to raise taxes soon."
    assert director._is_bio_mentioned(bio, out2.lower()) is True
    # Unrelated content — should NOT hit
    out3 = "The blacksmith was hammering on an anvil while a wolf howled."
    assert director._is_bio_mentioned(bio, out3.lower()) is False
    # Incidental one-word overlap — should NOT hit
    out4 = "The merchant is selling cloth today."
    assert director._is_bio_mentioned(bio, out4.lower()) is False
    print("  [PASS] is_bio_mentioned_detects_word_overlap")


def test_is_bio_mentioned_rejects_short_items():
    """Bio items with fewer than 3 content words can't support the
    overlap heuristic reliably — they should never count as mentioned
    to avoid false positives."""
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    assert director._is_bio_mentioned("Thin", "Thin") is False
    assert director._is_bio_mentioned("One two", "one two three") is False
    print("  [PASS] is_bio_mentioned_rejects_short_items")


def test_record_bio_mentions_bumps_count_on_match():
    """_record_bio_mentions should bump the count for a matched bio
    item and leave unrelated items alone."""
    restore = _isolate_state_file("bio_record")
    try:
        engine = _make_stub_engine()
        kael = engine.pie.npc_knowledge.get("kael")
        kael.personal_knowledge = [
            "I suspect Mara is selling counterfeit steel",
            "I lost my apprentice Tam in the forbidden forest last year",
        ]
        director = StoryDirector(engine)
        output = ("Kael shouts for Tam in the forbidden forest, hoping for "
                  "any sign of his lost apprentice.")
        director._record_bio_mentions("kael", output)
        counts = director._bio_mention_counts.get("kael", {})
        # The Tam item should have been matched
        tam_key = director._bio_item_key(
            "I lost my apprentice Tam in the forbidden forest last year"
        )
        steel_key = director._bio_item_key(
            "I suspect Mara is selling counterfeit steel"
        )
        assert counts.get(tam_key, 0) == 1, counts
        # The counterfeit steel item should NOT have been matched
        assert counts.get(steel_key, 0) == 0, counts
    finally:
        restore()
    print("  [PASS] record_bio_mentions_bumps_count_on_match")


def test_focus_npc_bio_rotates_heavily_mentioned_to_bottom():
    """When a goal has been mentioned at or above the cooldown
    threshold, _build_focus_npc_bio excludes it entirely until other
    goals catch up — hiding the over-used goal so the model literally
    can't see it. If excluded leaves the section empty, we fall back
    to showing the least-mentioned items so the bio never goes blank."""
    import npc_engine.story_director as sd_mod
    restore = _isolate_state_file("bio_rotate")
    try:
        engine = _make_stub_engine()
        kael = engine.pie.npc_knowledge.get("kael")
        kael.identity = {"name": "Kael", "role": "blacksmith",
                         "personality": "Gruff."}
        _attach_goals(engine, "kael", [
            {"id": "find_tam", "description": "Find my lost apprentice Tam in the forest", "priority": 10},
            {"id": "expose_mara", "description": "Prove Mara is selling fake steel", "priority": 6},
            {"id": "teach_pip", "description": "Teach Pip to become a smith", "priority": 3},
        ])
        director = StoryDirector(engine)

        # Fresh bio — highest priority should come first (p10 before p6).
        # Top-2 cap means p3 gets dropped from the display.
        bio1 = director._build_focus_npc_bio("kael")
        assert "[p10]" in bio1
        assert "[p6]" in bio1
        assert bio1.index("[p10]") < bio1.index("[p6]")

        # Now simulate heavy mention of the p10 goal. With cooldown
        # threshold = 2, a count of 5 means p10 gets EXCLUDED entirely
        # from the bio — the model shouldn't see it at all.
        tam_key = director._bio_item_key(
            "Find my lost apprentice Tam in the forest"
        )
        director._bio_mention_counts["kael"] = {tam_key: 5}

        bio2 = director._build_focus_npc_bio("kael")
        assert "[p10]" not in bio2, (
            "expected heavily-mentioned p10 goal to be excluded entirely\n"
            + bio2
        )
        # p6 and p3 should now both be eligible, with top-2 cap showing
        # both (priority order preserved within fresh items)
        assert "[p6]" in bio2
        assert "[p3]" in bio2
        assert bio2.index("[p6]") < bio2.index("[p3]"), (
            "fresh items should still respect priority order\n" + bio2
        )
    finally:
        restore()
    print("  [PASS] focus_npc_bio_rotates_heavily_mentioned_to_bottom")


def test_bio_cooldown_falls_back_when_all_items_exceed_threshold():
    """If every item in a section is above the cooldown threshold,
    _build_focus_npc_bio must fall back to showing the least-mentioned
    items rather than emitting an empty section."""
    import npc_engine.story_director as sd_mod
    restore = _isolate_state_file("bio_fallback")
    try:
        engine = _make_stub_engine()
        bess = engine.pie.npc_knowledge.get("bess")
        bess.identity = {"name": "Bess", "role": "innkeeper",
                         "personality": "Warm and gossipy."}
        bess.personal_knowledge = [
            "Overheard the merchant guild planning to raise taxes",
            "Knows a secret passage beneath the inn",
        ]
        director = StoryDirector(engine)

        # Bump both pk items past the cooldown threshold
        k1 = director._bio_item_key("Overheard the merchant guild planning to raise taxes")
        k2 = director._bio_item_key("Knows a secret passage beneath the inn")
        director._bio_mention_counts["bess"] = {k1: 5, k2: 3}

        bio = director._build_focus_npc_bio("bess")
        assert bio is not None
        # Fallback should still show both items — the LESS-mentioned
        # one (count 3) ahead of the more-mentioned (count 5)
        pk_section = bio.split("Private knowledge")[1]
        idx_secret = pk_section.index("secret passage")
        idx_guild = pk_section.index("merchant guild")
        assert idx_secret < idx_guild, (
            "fallback should order by ascending mention count\n" + bio
        )
    finally:
        restore()
    print("  [PASS] bio_cooldown_falls_back_when_all_items_exceed_threshold")


def test_focus_npc_bio_pk_rotates_by_mention_count():
    """Personal knowledge items should also rotate — least-mentioned
    first, list order as tiebreaker."""
    restore = _isolate_state_file("bio_pk_rotate")
    try:
        engine = _make_stub_engine()
        bess = engine.pie.npc_knowledge.get("bess")
        bess.identity = {"name": "Bess", "role": "innkeeper",
                         "personality": "Warm and gossipy."}
        bess.personal_knowledge = [
            "Overheard the merchant guild planning to raise taxes next season",
            "Knows a secret passage beneath the inn leading to cellar tunnels",
            "Her late husband built the inn twenty years ago",
            "Remembers the last tax collector who vanished on the north road",
        ]
        director = StoryDirector(engine)

        # Fresh bio — list order preserved
        bio1 = director._build_focus_npc_bio("bess")
        # First pk line should be the merchant guild one
        assert "merchant guild" in bio1.split("Private knowledge")[1].split("\n")[1]

        # Bump the merchant guild mention to 3
        guild_key = director._bio_item_key(bess.personal_knowledge[0])
        director._bio_mention_counts["bess"] = {guild_key: 3}

        bio2 = director._build_focus_npc_bio("bess")
        # The merchant guild item should now be LAST in the pk list
        # (or dropped if top-4 truncation kicks in)
        pk_section = bio2.split("Private knowledge")[1]
        pk_lines = [ln for ln in pk_section.split("\n") if ln.strip().startswith("-")]
        assert pk_lines, pk_section
        first_pk = pk_lines[0]
        assert "merchant guild" not in first_pk, (
            "expected heavily-mentioned merchant guild to be demoted\n" + bio2
        )
    finally:
        restore()
    print("  [PASS] focus_npc_bio_pk_rotates_by_mention_count")


def test_focus_npc_bio_header_contains_paraphrase_instruction():
    """The bio header must explicitly tell the model to paraphrase —
    this counters the verbatim phrasing clone we saw in the rotation
    run (Bess's 'merchant guild planning to raise taxes' near-quote)."""
    engine = _make_stub_engine()
    kael = engine.pie.npc_knowledge.get("kael")
    kael.identity = {"name": "Kael", "role": "blacksmith",
                     "personality": "Gruff."}
    kael.personal_knowledge = ["I lost my apprentice Tam in the Silverwood."]
    director = StoryDirector(engine)
    bio = director._build_focus_npc_bio("kael")
    assert bio is not None
    assert "PARAPHRASE" in bio.upper(), bio
    print("  [PASS] focus_npc_bio_header_contains_paraphrase_instruction")


def test_bio_mention_counts_persist_across_instances():
    """Mention counts should round-trip through state.json so the
    rotation effect survives a director restart."""
    restore = _isolate_state_file("bio_persist")
    try:
        engine = _make_stub_engine()
        director1 = StoryDirector(engine)
        director1._bio_mention_counts = {
            "kael": {"find tam in the forest": 2, "expose mara": 1},
            "bess": {"merchant guild taxes": 3},
        }
        director1._save_state()

        director2 = StoryDirector(engine)
        assert director2._bio_mention_counts == director1._bio_mention_counts
    finally:
        restore()
    print("  [PASS] bio_mention_counts_persist_across_instances")


# ── Example rotation tests ──────────────────────────────────────

def test_pick_examples_excludes_focus_npc():
    """When focus_npc matches a primary_npc in the library, that example
    must be excluded from the picks. This breaks the copy-loop where
    the model rewrites its own focus NPC's example verbatim."""
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    picks = director._pick_examples(focus_npc="kael", action_kind="event")
    primary_npcs = [p.get("primary_npc") for p in picks]
    assert "kael" not in primary_npcs, f"kael example should be excluded: {primary_npcs}"
    # And picks is non-empty (fallback shouldn't kick in — we have 4 eligible)
    assert len(picks) > 0
    print("  [PASS] pick_examples_excludes_focus_npc")


def test_pick_examples_prefers_target_action_kind():
    """The picker must include at least one example whose action kind
    matches the target kind — this stabilizes the schema for the shape
    the worker is about to emit."""
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    picks = director._pick_examples(focus_npc="noah", action_kind="quest")
    picked_kinds = [p.get("action", {}).get("action") for p in picks]
    assert "quest" in picked_kinds, f"quest kind missing: {picked_kinds}"
    # And we should get at most 3
    assert len(picks) <= 3
    print("  [PASS] pick_examples_prefers_target_action_kind")


def test_pick_examples_caps_at_three():
    """Even if every example is eligible, the picker returns at most 3
    to keep the prompt lean — fewer examples gives bio and arc blocks
    more salience."""
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    # Focus on an NPC that's not primary in any example — everything qualifies
    picks = director._pick_examples(focus_npc="mara", action_kind="fact")
    assert len(picks) <= 3, f"expected at most 3 picks, got {len(picks)}"
    print("  [PASS] pick_examples_caps_at_three")


def test_pick_examples_terse_mode_caps_at_two():
    """Terse mode trims the example budget from 3 picks to 2 — terse
    examples are short enough that 2 cover target + alternate shape,
    and the extra pick was pure prompt bloat."""
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    director.narration_mode = "terse"
    picks = director._pick_examples(focus_npc="mara", action_kind="fact")
    assert len(picks) <= 2, f"expected at most 2 picks in terse mode, got {len(picks)}"
    print("  [PASS] pick_examples_terse_mode_caps_at_two")


def test_pick_examples_falls_back_when_all_match_focus():
    """If the entire library is about the focus NPC (pathological but
    possible with a shrunk library), the picker returns all of them
    rather than an empty list — an empty EXAMPLES block tanks schema
    parse reliability on small models."""
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    # Manually replace the loaded examples with an all-kael library
    director._examples = [
        {"primary_npc": "kael", "world_state": "kael does something",
         "action": {"action": "event", "target": "kael", "event": "x"}},
        {"primary_npc": "kael", "world_state": "kael does something else",
         "action": {"action": "fact", "npc_id": "kael", "fact": "y"}},
    ]
    picks = director._pick_examples(focus_npc="kael", action_kind="event")
    assert len(picks) == 2, f"fallback should return all available: {picks}"
    print("  [PASS] pick_examples_falls_back_when_all_match_focus")


def test_pick_examples_diversifies_action_kinds():
    """When filling remaining picks, the picker prefers examples of
    kinds it hasn't shown yet — so the worker sees all three shapes."""
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    picks = director._pick_examples(focus_npc="noah", action_kind="quest")
    kinds = [p.get("action", {}).get("action") for p in picks]
    # With 5 real examples and focus=noah (not a primary), we should
    # get 3 different kinds
    assert len(set(kinds)) >= 2, f"expected kind diversity: {kinds}"
    print("  [PASS] pick_examples_diversifies_action_kinds")


def test_prompt_excludes_focus_npc_examples():
    """_build_prompt should only include examples about NON-focus NPCs
    so the forced-focus directive gets to drive the content without a
    per-NPC example template fighting it."""
    engine = _make_stub_engine()
    director = StoryDirector(engine)
    prompt = director._build_prompt("world snap", focus_npc="kael",
                                     action_kind="event")
    # The kael missing_hammers example mentions that specific quest id
    assert "missing_hammers" not in prompt, (
        "kael's missing_hammers example should not appear in a "
        "kael-focused prompt:\n" + prompt
    )
    # But OTHER examples should be there — e.g., the pip shipment
    # example or the bess tax-collector-body example
    assert ("tax_collector_body" in prompt or
            "Silverwood" in prompt or
            "floorboards" in prompt), (
        "expected at least one non-kael example to appear:\n" + prompt
    )
    print("  [PASS] prompt_excludes_focus_npc_examples")


# ── Runner ──────────────────────────────────────────────────────

def main():
    print("Story Director — offline unit tests")
    test_parse_clean_json()
    test_parse_json_with_fences_and_prose()
    test_parse_garbage_returns_noop()
    test_parse_coerces_mislabeled_event_to_fact()
    test_parse_coerces_mislabeled_quest_payload()
    test_parse_extracts_first_of_two_json_objects()
    test_parse_noop_strips_hallucinated_fields()
    test_focus_npc_picks_untouched_first()
    test_focus_npc_rotates_by_least_recently_touched()
    test_enforce_focus_npc_overrides_event_target()
    test_enforce_focus_npc_overrides_quest_npc_id()
    test_enforce_focus_npc_leaves_noop_alone()
    test_prompt_contains_focus_block()
    test_action_kind_rotates_over_session()
    test_action_kind_skips_quest_when_npc_full()
    test_enforce_action_kind_event_to_quest()
    test_enforce_action_kind_leaves_noop_alone()
    test_enforce_action_kind_event_to_fact()
    test_architect_plan_picks_distinct_npcs()
    test_architect_plan_caps_at_npc_count()
    test_burst_rotation_holds_slot_zero_across_k_ticks()
    test_burst_rotation_coexists_with_multi_action_rotation()
    test_burst_rotation_persists_across_director_instances()
    test_burst_rotation_recovers_when_focus_npc_vanishes()
    test_pick_focus_respects_extra_exclude()
    test_multi_action_tick_runs_n_workers()
    test_small_cast_caps_actions_per_tick_to_one()
    test_large_cast_honors_requested_actions_per_tick()
    test_small_cast_cap_threshold_boundary()
    test_small_cast_single_action_passthrough()

    print("\nStory Director — zone layer tests (Phase 1)")
    test_empty_active_zones_preserves_world_wide_mode()
    test_zone_soft_weighting_prefers_in_zone()
    test_zone_out_of_zone_rate_honored_precisely()
    test_global_zone_npc_always_in_scope()
    test_quest_hard_filter_on_out_of_zone_focus()
    test_bounded_snapshot_surfaces_active_zone_npcs()
    test_npc_current_zone_mutation_persists()
    test_immobile_npc_zone_change_rejected()
    test_zone_change_breaks_burst_rotation()
    test_burst_rotation_locks_to_in_zone()
    test_pending_player_target_bypasses_zone_filter()

    print("\nStory Director — lifecycle: death plumbing (Phase 2a)")
    test_queue_death_request_validates_alive_and_exists()
    test_death_dispatch_marks_npc_deceased_and_breaks_burst()
    test_deceased_npc_excluded_from_focus_selection()
    test_death_transfers_open_quests_to_inheritor()
    test_death_aborts_open_quests_without_inheritor()
    test_arc_on_cast_death_resolves_post_confront_arc()
    test_arc_on_cast_death_trims_cast_on_early_arc()
    test_lifecycle_tick_drains_pending_death()
    test_recently_departed_section_in_snapshot()
    test_postgen_detect_deceased_reference_catches_living_action()
    test_postgen_repair_deceased_reference_adds_late_prefix()

    print("\nStory Director — lifecycle: birth pipeline (Phase 2b)")
    test_birth_dispatch_creates_profile_and_registers()
    test_population_gap_triggers_birth_on_lifecycle_tick()
    test_population_at_target_no_birth()
    test_queue_birth_request_queues_and_drains()
    test_birth_name_avoids_collision_with_existing_npcs()

    print("\nStory Director — autonomous lifecycle (Phase 2c)")
    test_autonomous_off_no_director_deaths()
    test_autonomous_on_proposes_death_at_confront()
    test_autonomous_death_capped_by_session_limit()
    test_autonomous_skips_death_when_no_arc_at_confront()
    test_player_killed_npc_full_propagation()

    test_single_action_tick_is_backward_compatible()
    test_dialogue_autofeed_format()
    test_record_player_action_and_snapshot()
    test_focus_npc_prioritizes_pending_player_target()
    test_pending_player_target_ignores_stale_actions()
    test_record_player_action_routes_quest_completion_to_engine()
    test_record_player_action_routes_quest_acceptance_to_engine()
    test_fact_ledger_flags_similar_text()
    test_fact_ledger_does_not_flag_unrelated_text()
    test_fact_ledger_persists_and_reloads()
    test_ledger_sidecar_round_trips_embedding_values()
    test_ledger_loads_legacy_inline_format()
    test_ledger_sidecar_is_smaller_than_inline_json()
    test_contradiction_checker_catches_known_contradiction()
    test_contradiction_checker_does_not_flag_paraphrase()
    test_contradiction_checker_does_not_flag_plot_escalation()
    test_contradiction_checker_silent_when_unavailable()
    test_fact_ledger_warning_includes_nli_when_model_available()
    test_contradiction_retry_redoes_worker_when_pre_check_fires()
    test_no_retry_when_pre_check_finds_no_contradiction()
    test_fact_ledger_check_separates_from_add()
    test_fact_ledger_silent_when_embedder_missing()
    test_cross_session_persistence_smoke()
    test_player_actions_trimmed_to_last_8()
    test_dispatch_quest_adds_to_npc()
    test_dispatch_quest_dedupes_by_id()
    test_dispatch_event_global_vs_targeted()
    test_dispatch_fact_world_and_personal()
    test_full_tick_loop_with_stubbed_model()
    test_tick_persists_state_between_instances()

    print("\nStory Director — narrative arc tests")
    test_arc_planner_proposes_from_clustered_ledger()
    test_arc_planner_skips_with_too_few_entries()
    test_arc_planner_caps_at_max_concurrent_arcs()
    test_arc_planner_proposal_excludes_npcs_in_active_casts()
    test_arc_for_focus_returns_matching_arc()
    test_arc_for_focus_prefers_weakest_thread_on_overlap()
    test_record_cast_touch_bumps_all_matching_arcs()
    test_advance_if_beat_met_advances_multiple_arcs()
    test_arc_planner_advances_beat_after_n_touches()
    test_arc_planner_record_cast_touch_bumps_counter()
    test_arc_planner_record_cast_touch_ignores_inactive_arcs()
    test_arc_planner_resolves_after_all_beats()
    test_prompt_contains_active_arc_block()
    test_arc_planner_persists_and_reloads()
    test_arc_planner_cooldown_prevents_immediate_reproposal()

    print("\nStory Director — NPC bio injection tests")
    test_peek_npc_goals_reads_priority_sorted()
    test_focus_npc_bio_contains_personality_goals_and_knowledge()
    test_focus_npc_bio_returns_none_when_npc_is_bare()
    test_world_snapshot_roster_includes_top_goal()
    test_prompt_contains_bio_block_for_focus_npc()
    test_prompt_skips_bio_block_when_npc_is_bare()

    print("\nStory Director — self-repetition precheck tests")
    test_precheck_self_repetition_fires_on_high_sim_same_npc_recent()
    test_precheck_self_repetition_restricts_to_same_npc()
    test_ledger_check_filters_by_restrict_to_npc()
    test_precheck_self_repetition_ignores_low_similarity()
    test_precheck_self_repetition_ignores_old_matches()
    test_precheck_self_repetition_ignores_all_target()
    test_self_repetition_retry_dispatches_second_response()
    test_self_repetition_retry_falls_back_to_original_on_noop()
    test_contradiction_takes_precedence_over_self_repetition()

    print("\nStory Director — narration mode tests")
    test_narration_mode_defaults_to_prose()
    test_narration_mode_terse_injects_output_style_block()
    test_set_narration_mode_reloads_examples_terse_library()
    test_set_narration_mode_rejects_unknown_mode()

    print("\nStory Director — self-rep retry budget tests")
    test_self_rep_budget_resets_each_tick()
    test_self_rep_budget_blocks_second_retry_in_same_tick()
    test_contradiction_retry_not_budget_gated()

    print("\nStory Director — intra-bio rotation tests")
    test_is_bio_mentioned_detects_word_overlap()
    test_is_bio_mentioned_rejects_short_items()
    test_record_bio_mentions_bumps_count_on_match()
    test_focus_npc_bio_rotates_heavily_mentioned_to_bottom()
    test_bio_cooldown_falls_back_when_all_items_exceed_threshold()
    test_focus_npc_bio_pk_rotates_by_mention_count()
    test_focus_npc_bio_header_contains_paraphrase_instruction()
    test_bio_mention_counts_persist_across_instances()

    print("\nStory Director — example rotation tests")
    test_pick_examples_excludes_focus_npc()
    test_pick_examples_prefers_target_action_kind()
    test_pick_examples_caps_at_three()
    test_pick_examples_terse_mode_caps_at_two()
    test_pick_examples_falls_back_when_all_match_focus()
    test_pick_examples_diversifies_action_kinds()
    test_prompt_excludes_focus_npc_examples()

    print("\nStory Director — integration smoke test")
    test_integration_tick_mutates_world()

    print("\nAll story director tests passed.")


if __name__ == "__main__":
    main()
