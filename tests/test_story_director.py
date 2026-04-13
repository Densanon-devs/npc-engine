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
)


# ── Offline stub engine ─────────────────────────────────────────

class _StubNPC:
    """Tiny stand-in for NPCKnowledge that supports add_quest + fields the
    snapshot builder pokes at."""

    def __init__(self, npc_id: str, role: str):
        self.identity = {"name": npc_id.title(), "role": role}
        self.world_facts: list[str] = []
        self.personal_knowledge: list[str] = []
        self.quests: list[Quest] = []
        self.events: list[SimpleNamespace] = []

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
            npc.personal_knowledge.append(fact)
        else:
            npc.world_facts.append(fact)
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


def _make_stub_engine(responses=None):
    profiles = {
        "kael": _StubNPC("kael", "blacksmith"),
        "bess": _StubNPC("bess", "innkeeper"),
        "noah": _StubNPC("noah", "elder"),
    }
    engine = _StubEngine(profiles)
    engine.pie.base_model = _StubBaseModel(responses or [])
    return engine


def _isolate_state_file(tag: str):
    """Redirect StoryDirector's STATE_FILE *and* LEDGER_FILE to per-test
    temp paths so offline tests don't scribble on the real runtime
    files. Returns a restore fn that puts the originals back and
    deletes the temp files."""
    import npc_engine.story_director as sd_mod
    original_state = sd_mod.STATE_FILE
    original_ledger = sd_mod.LEDGER_FILE
    tmp_state = NPC_ROOT / "data" / "story_director" / f"_tmp_{tag}_state.json"
    tmp_ledger = NPC_ROOT / "data" / "story_director" / f"_tmp_{tag}_ledger.json"
    for p in (tmp_state, tmp_ledger):
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass
    sd_mod.STATE_FILE = tmp_state
    sd_mod.LEDGER_FILE = tmp_ledger

    def restore():
        sd_mod.STATE_FILE = original_state
        sd_mod.LEDGER_FILE = original_ledger
        for p in (tmp_state, tmp_ledger):
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
    tick, just with parallel workers)."""
    restore = _isolate_state_file("multi")
    try:
        engine = _make_stub_engine(responses=[
            '{"action": "event", "target": "kael", "event": "kael event"}',
            '{"action": "event", "target": "bess", "event": "bess event"}',
            '{"action": "event", "target": "noah", "event": "noah event"}',
        ])
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
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        ledger = FactLedger(tmp_path)
        if ledger.embedder is None:
            print("  [SKIP] fact_ledger_persists_and_reloads — embedder unavailable")
            return
        ledger.add("Bess hears a strange whisper at the inn.",
                   npc_id="bess", kind="event", tick=1)
        assert tmp_path.exists()
        ledger2 = FactLedger(tmp_path)
        assert len(ledger2.entries) == 1, ledger2.entries
        assert ledger2.entries[0]["npc_id"] == "bess"
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    print("  [PASS] fact_ledger_persists_and_reloads")


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
    assert "The tax collector was found dead." in noah.world_facts
    assert "I fear the king's letter." in noah.personal_knowledge
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
    pre_facts = {npc_id: (len(npc.world_facts) + len(npc.personal_knowledge))
                 for npc_id, npc in engine.pie.npc_knowledge.profiles.items()}

    result = engine.story_director.tick(max_tokens=300)
    print(f"    tick result action: {result['action'].get('action')}")
    print(f"    tick result dispatch: {result['dispatch']}")
    print(f"    raw preview: {str(result['raw_response'])[:200]}")

    post_quests = {npc_id: len(npc.quests)
                   for npc_id, npc in engine.pie.npc_knowledge.profiles.items()}
    post_events = {npc_id: len(npc.events)
                   for npc_id, npc in engine.pie.npc_knowledge.profiles.items()}
    post_facts = {npc_id: (len(npc.world_facts) + len(npc.personal_knowledge))
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
    test_pick_focus_respects_extra_exclude()
    test_multi_action_tick_runs_n_workers()
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

    print("\nStory Director — integration smoke test")
    test_integration_tick_mutates_world()

    print("\nAll story director tests passed.")


if __name__ == "__main__":
    main()
