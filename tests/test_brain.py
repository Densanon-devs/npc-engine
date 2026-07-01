"""Anima brain over the wire, with a MOCK NPCEngine (no model needed).
Run with: python test_brain_mock.py"""
import io, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from npc_engine.brain import AnimaBrain, handle, serve_stdio, COMMANDS

passed = failed = 0
def check(name, fn):
    global passed, failed
    try:
        fn(); passed += 1; print(f"  PASS  {name}")
    except Exception as e:
        failed += 1; print(f"  FAIL  {name}: {type(e).__name__}: {e}")

class MockNPCEngine:
    def __init__(self): self.mood=None; self.events=[]
    def initialize(self): pass
    def process(self, text, npc_id=None): return f"[{npc_id or 'npc'}] heard: {text}"
    def inject_event(self, desc, npc_id=None): self.events.append(desc)
    def switch_npc(self, npc_id): return {"active": npc_id}
    def set_mood(self, npc_id, mood, intensity=0.5): self.mood=mood; return {"npc_id":npc_id,"mood":mood}
    def adjust_trust(self, npc_id, delta, reason=""): return {"delta": delta}
    def add_knowledge(self, npc_id, fact): return {"added": fact}
    def accept_quest(self, qid, name, given_by): return {"quest": qid}
    def complete_quest(self, qid): return {"done": qid}
    def list_npcs(self): return [{"id":"noah","name":"Noah"}]
    def active_npc(self): return "noah"
    def get_npc_state(self, npc_id): return {"mood": self.mood}
    def get_social_graph(self): return {"noah": {}}
    def shutdown(self): pass

def _brain(): return AnimaBrain(engine=MockNPCEngine())

def test_reset_state():
    b=_brain(); st=b.reset()
    assert st["npcs"] and st["active"]=="noah"

def test_talk_returns_reply():
    b=_brain(); b.reset()
    r=b.command("talk", kwargs={"text":"hello","npc_id":"noah"})
    assert "heard: hello" in r

def test_capabilities_dispatch():
    b=_brain(); b.reset()
    assert b.command("set_mood", kwargs={"npc_id":"noah","mood":"angry"})["mood"]=="angry"
    assert b.command("adjust_trust", kwargs={"npc_id":"noah","delta":5})["delta"]==5
    b.command("event", kwargs={"description":"door slammed"})
    assert b._engine.events==["door slammed"]

def test_rpc_and_json():
    b=_brain()
    assert handle(b, {"op":"reset"})["ok"]
    assert handle(b, {"op":"info"})["info"]["engine"]=="anima"
    r=handle(b, {"op":"command","name":"talk","kwargs":{"text":"hi"}})
    assert r["ok"] and "heard: hi" in r["result"]
    json.dumps(handle(b, {"op":"state"}))
    assert handle(b, {"op":"command","name":"rm_rf"})["ok"] is False   # disallowed

def test_stdio_roundtrip():
    b=_brain()
    req="\n".join([json.dumps({"op":"reset"}),
                   json.dumps({"op":"command","name":"talk","kwargs":{"text":"yo"}}),
                   json.dumps({"op":"shutdown"})])
    out=io.StringIO(); serve_stdio(b, io.StringIO(req), out)
    lines=[json.loads(x) for x in out.getvalue().splitlines() if x.strip()]
    assert all(r["ok"] for r in lines) and lines[-1].get("shutdown")

check("Anima.reset_state", test_reset_state)
check("Anima.talk_reply", test_talk_returns_reply)
check("Anima.capabilities", test_capabilities_dispatch)
check("Anima.rpc_json", test_rpc_and_json)
check("Anima.stdio_roundtrip", test_stdio_roundtrip)
print(f"\n{'='*50}\nResults: {passed}/{passed+failed} passed")
sys.exit(0 if failed==0 else 1)
