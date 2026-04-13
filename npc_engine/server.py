"""
NPC Engine — REST API Server.

Game integration endpoints for NPC dialogue, social graph,
gossip propagation, and capability state.

Usage:
    python -m npc_engine.server
    python -m npc_engine.server --port 9000
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger("NPCEngine.server")

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        import os
        config_path = os.environ.get("NPC_ENGINE_CONFIG", "config.yaml")
        from npc_engine.engine import NPCEngine
        _engine = NPCEngine(config_path)
        _engine.initialize()
        logger.info("NPC Engine initialized for API server")
    return _engine


def create_app():
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
        from typing import Optional
    except ImportError:
        print("FastAPI not installed. Install with: pip install fastapi uvicorn")
        sys.exit(1)

    app = FastAPI(
        title="Anima",
        description="Every NPC has a soul. Game integration API for NPC dialogue with gossip, trust, and capabilities.",
        version="0.2.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request Models ───────────────────────────────────────

    class GenerateRequest(BaseModel):
        prompt: str
        npc_id: Optional[str] = None
        max_tokens: Optional[int] = None
        temperature: Optional[float] = None

    class SwitchRequest(BaseModel):
        npc_id: str

    class EventRequest(BaseModel):
        description: str
        npc_id: Optional[str] = None

    # ── Endpoints ────────────────────────────────────────────

    @app.on_event("startup")
    async def startup():
        get_engine()

    @app.post("/generate")
    async def generate(req: GenerateRequest):
        """Generate NPC dialogue. Optionally specify npc_id to switch first."""
        engine = get_engine()

        if req.max_tokens:
            engine.pie.config.base_model.max_tokens = req.max_tokens
        if req.temperature:
            engine.pie.config.base_model.temperature = req.temperature

        start = time.monotonic()
        response = engine.process(req.prompt, npc_id=req.npc_id)
        elapsed = time.monotonic() - start

        npc_id = req.npc_id or engine.active_npc
        cap_state = engine.get_npc_state(npc_id).get("shared_state", {})

        return {
            "npc_id": npc_id,
            "response": response,
            "generation_time": round(elapsed, 3),
            "capability_state": cap_state,
        }

    @app.get("/npc/list")
    async def npc_list():
        """List all NPCs with capabilities."""
        engine = get_engine()
        return {
            "active": engine.active_npc,
            "world_name": engine.config.world_name,
            "npcs": engine.list_npcs(),
        }

    @app.post("/npc/switch")
    async def npc_switch(req: SwitchRequest):
        """Switch active NPC."""
        engine = get_engine()
        try:
            return engine.switch_npc(req.npc_id)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/npc/{npc_id}/state")
    async def npc_state(npc_id: str):
        """Get NPC capability state."""
        engine = get_engine()
        return engine.get_npc_state(npc_id)

    @app.post("/events/inject")
    async def inject_event(req: EventRequest):
        """Inject world event to one or all NPCs."""
        engine = get_engine()
        engine.inject_event(req.description, npc_id=req.npc_id)
        return {
            "injected": True,
            "target": req.npc_id or "all",
            "event": req.description,
        }

    @app.get("/gossip/graph")
    async def gossip_graph():
        """Get social graph data."""
        engine = get_engine()
        return engine.get_social_graph()

    @app.get("/gossip/{npc_id}")
    async def npc_gossip(npc_id: str):
        """Get rumors an NPC has heard."""
        engine = get_engine()
        state = engine.get_npc_state(npc_id)
        gossip = state.get("capabilities", {}).get("gossip", {})
        return {
            "npc_id": npc_id,
            "rumors": gossip.get("rumors", []),
        }

    @app.get("/quests")
    async def quest_state():
        """Get player quest state."""
        engine = get_engine()
        pq = engine.pie.player_quests
        return {
            "player_name": pq.player_name,
            "active": pq.active_quests,
            "completed": pq.completed_quests,
            "reputation": pq.reputation,
        }

    # ── Game State Mutation Endpoints ────────────────────────

    class QuestAcceptRequest(BaseModel):
        quest_id: str
        quest_name: str
        given_by: str               # NPC ID that gives the quest

    class QuestCompleteRequest(BaseModel):
        quest_id: str

    class TrustAdjustRequest(BaseModel):
        npc_id: str
        delta: int                  # Positive or negative
        reason: Optional[str] = ""  # e.g., "player gave gift", "player stole"

    class ScratchpadRequest(BaseModel):
        npc_id: str
        text: str                   # Fact to remember
        importance: Optional[float] = 0.7  # 0.0-1.0

    class MoodRequest(BaseModel):
        npc_id: str
        mood: str                   # e.g., "angry", "happy", "fearful"
        intensity: Optional[float] = 0.5
        pin_turns: Optional[int] = 3  # how many turns the mood resists model override

    class KnowledgeRequest(BaseModel):
        npc_id: str
        fact: str
        fact_type: Optional[str] = "world"  # "world" or "personal"

    class UnlockGateRequest(BaseModel):
        npc_id: str
        gate_id: str                # ID from the NPC's knowledge_gate config

    class StoryTickRequest(BaseModel):
        max_tokens: Optional[int] = 400
        temperature: Optional[float] = 0.7

    class PlayerActionRequest(BaseModel):
        text: str
        target: Optional[str] = None
        trust_delta: Optional[int] = None

    @app.post("/quests/accept")
    async def quest_accept(req: QuestAcceptRequest):
        """Player accepts a quest. Updates tracker and NPC state."""
        engine = get_engine()
        return engine.accept_quest(req.quest_id, req.quest_name, req.given_by)

    @app.post("/quests/complete")
    async def quest_complete(req: QuestCompleteRequest):
        """Player completes a quest. Triggers trust boost and gossip propagation."""
        engine = get_engine()
        result = engine.complete_quest(req.quest_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result

    @app.post("/npc/trust")
    async def adjust_trust(req: TrustAdjustRequest):
        """Directly adjust an NPC's trust level. For game-triggered events."""
        engine = get_engine()
        result = engine.adjust_trust(req.npc_id, req.delta, req.reason)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result

    @app.post("/npc/scratchpad")
    async def add_scratchpad(req: ScratchpadRequest):
        """Add a fact to an NPC's scratchpad memory."""
        engine = get_engine()
        result = engine.add_scratchpad_entry(req.npc_id, req.text, req.importance)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result

    @app.post("/npc/mood")
    async def set_mood(req: MoodRequest):
        """Set an NPC's mood directly. For cutscenes, world events, etc."""
        engine = get_engine()
        result = engine.set_mood(req.npc_id, req.mood, req.intensity, pin_turns=req.pin_turns)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result

    @app.post("/npc/knowledge")
    async def add_knowledge(req: KnowledgeRequest):
        """Add a fact to an NPC's knowledge at runtime."""
        engine = get_engine()
        result = engine.add_knowledge(req.npc_id, req.fact, req.fact_type)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result

    @app.post("/npc/unlock-gate")
    async def unlock_gate(req: UnlockGateRequest):
        """Force-unlock a gated knowledge fact. For quest rewards, story progression."""
        engine = get_engine()
        result = engine.unlock_knowledge_gate(req.npc_id, req.gate_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result

    # ── Story Director (world-level overseer) ───────────────

    @app.post("/story/tick")
    async def story_tick(req: StoryTickRequest):
        """Advance the world story by one overseer decision."""
        engine = get_engine()
        if engine.story_director is None:
            raise HTTPException(status_code=503, detail="Story Director not initialized")
        return engine.story_director.tick(
            max_tokens=req.max_tokens or 400,
            temperature=req.temperature if req.temperature is not None else 0.7,
        )

    @app.get("/story/state")
    async def story_state():
        """Get the Story Director's current state (tick count, recent decisions)."""
        engine = get_engine()
        if engine.story_director is None:
            raise HTTPException(status_code=503, detail="Story Director not initialized")
        return engine.story_director.get_state()

    @app.post("/story/player_action")
    async def story_player_action(req: PlayerActionRequest):
        """Record something the player did so the Director reacts next tick."""
        engine = get_engine()
        if engine.story_director is None:
            raise HTTPException(status_code=503, detail="Story Director not initialized")
        result = engine.story_director.record_player_action(
            req.text, target=req.target, trust_delta=req.trust_delta,
        )
        if not result.get("ok"):
            raise HTTPException(status_code=400, detail=result.get("reason", "bad_request"))
        return result

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    return app


def main():
    parser = argparse.ArgumentParser(description="NPC Engine — REST API Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", "-p", type=int, default=8000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed. Install with: pip install uvicorn")
        sys.exit(1)

    print(f"\n  Anima — Every NPC has a soul")
    print(f"  http://{args.host}:{args.port}")
    print(f"  Docs: http://{args.host}:{args.port}/docs\n")

    uvicorn.run("npc_engine.server:create_app", host=args.host, port=args.port, factory=True)


if __name__ == "__main__":
    main()
