"""
Anima as a densanon "brain" — expose the NPC engine over the same newline-
delimited JSON-RPC stdio wire that densanon's game brains use, so any face
(the densanon editor, Unity, Unreal) drives Anima identically.

    python -m npc_engine.brain [config.yaml]

Self-contained: Anima is its own product, so this implements the (small, stable)
brain protocol directly rather than depending on densanon-core. The protocol:

  {"op":"reset"}                                  -> {"ok":true,"state":{...}}
  {"op":"command","name":"talk","kwargs":{...}}   -> {"ok":true,"result":...}
  {"op":"observe","commands":[...]}               -> {"ok":true}
  {"op":"step"}                                   -> {"ok":true}      # no-op (dialogue is event-driven)
  {"op":"state"}                                  -> {"ok":true,"state":{...}}
  {"op":"info"} / {"op":"shutdown"}
errors -> {"ok":false,"error":"..."}
"""
from __future__ import annotations

import json
import sys
from typing import Any


# ── command surface: the observations/intents a face may send ────
# name -> handler(engine, **kwargs) -> json-able result
def _talk(engine, text, npc_id=None):
    return engine.process(str(text), npc_id)

def _event(engine, description, npc_id=None):
    engine.inject_event(str(description), npc_id); return True

def _switch(engine, npc_id):
    return engine.switch_npc(str(npc_id))

def _mood(engine, npc_id, mood, intensity=0.5):
    return engine.set_mood(str(npc_id), str(mood), float(intensity))

def _trust(engine, npc_id, delta, reason=""):
    return engine.adjust_trust(str(npc_id), int(delta), str(reason))

def _knowledge(engine, npc_id, fact):
    return engine.add_knowledge(str(npc_id), str(fact))

def _accept_quest(engine, quest_id, quest_name, given_by):
    return engine.accept_quest(str(quest_id), str(quest_name), str(given_by))

def _complete_quest(engine, quest_id):
    return engine.complete_quest(str(quest_id))

def _add_profile(engine, profile_path):
    return engine.add_profile(str(profile_path))


COMMANDS = {
    "talk": _talk,                    # the core: player text -> NPC reply
    "event": _event,                  # world event raises awareness/context
    "switch_npc": _switch,
    "set_mood": _mood,
    "adjust_trust": _trust,
    "add_knowledge": _knowledge,
    "accept_quest": _accept_quest,
    "complete_quest": _complete_quest,
    "add_profile": _add_profile,
}


def _jsonable(obj: Any):
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            return _jsonable(to_dict())
        except Exception:
            pass
    return str(obj)


class AnimaBrain:
    """Wraps an NPCEngine as a brain. Pass an engine for tests; otherwise it
    builds NPCEngine(config_path) on reset."""

    def __init__(self, engine=None, config_path: str = "config.yaml"):
        self._engine = engine
        self._config_path = config_path

    def reset(self, setup=None) -> dict:
        if self._engine is None:
            from npc_engine.engine import NPCEngine
            self._engine = NPCEngine(self._config_path)
        init = getattr(self._engine, "initialize", None)
        if callable(init):
            init()
        for cmd in (setup or []):
            self.command(cmd["name"], cmd.get("args"), cmd.get("kwargs"))
        return self.state()

    def command(self, name, args=None, kwargs=None):
        if self._engine is None:
            raise RuntimeError("brain not reset")
        handler = COMMANDS.get(name)
        if handler is None:
            raise RuntimeError(f"command {name!r} not allowed")
        return _jsonable(handler(self._engine, *(args or []), **(kwargs or {})))

    def step(self, dt: float = 0.0, ticks: int = 1) -> None:
        # Dialogue is event-driven (talk/event); there's no fixed-dt sim to tick.
        return None

    def state(self) -> dict:
        e = self._engine
        npcs = _jsonable(e.list_npcs()) if hasattr(e, "list_npcs") else []
        active = getattr(e, "active_npc", None)     # @property or method
        if callable(active):
            active = active()
        social = _jsonable(e.get_social_graph()) if hasattr(e, "get_social_graph") else {}
        return {"npcs": npcs, "active": _jsonable(active), "social": social}

    def info(self) -> dict:
        return {"engine": "anima", "summary": "local NPC dialogue AI",
                "commands": sorted(COMMANDS)}


def handle(brain: AnimaBrain, req: dict) -> dict:
    op = req.get("op")
    try:
        if op == "info":
            return {"ok": True, "info": brain.info()}
        if op == "reset":
            return {"ok": True, "state": brain.reset(req.get("setup"))}
        if op == "command":
            return {"ok": True,
                    "result": brain.command(req["name"], req.get("args"),
                                            req.get("kwargs"))}
        if op == "observe":
            for c in req.get("commands", []):
                brain.command(c["name"], c.get("args"), c.get("kwargs"))
            return {"ok": True}
        if op == "step":
            brain.step(float(req.get("dt", 0.0)), int(req.get("ticks", 1)))
            return {"ok": True}
        if op == "state":
            return {"ok": True, "state": brain.state()}
        if op == "shutdown":
            sd = getattr(brain._engine, "shutdown", None)
            if callable(sd):
                sd()
            return {"ok": True, "shutdown": True}
        return {"ok": False, "error": f"unknown op {op!r}"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def serve_stdio(brain: AnimaBrain, stdin, stdout) -> None:
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            stdout.write(json.dumps({"ok": False, "error": f"bad json: {e}"}) + "\n")
            stdout.flush()
            continue
        resp = handle(brain, req)
        stdout.write(json.dumps(resp) + "\n")
        stdout.flush()
        if resp.get("shutdown"):
            break


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    config = argv[0] if argv else "config.yaml"
    # Keep the protocol channel PURE: model-load / PIE / llama_cpp noise must not
    # land on the stdout the client parses. Reserve the real stdout for JSON only
    # and send everything else (any print()) to stderr.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    serve_stdio(AnimaBrain(config_path=config), sys.stdin, real_stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
