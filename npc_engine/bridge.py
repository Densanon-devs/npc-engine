"""
Bridge — imports from densanon-core and PIE.

NPC Engine depends on:
  - densanon-core: reusable building blocks (model loader, expert system, etc.)
  - PIE: the orchestrator that wires core modules together

In development: both are sibling directories.
In PyInstaller bundle: everything is on sys.path already.
"""

import sys
from pathlib import Path

_NPC_ENGINE_ROOT = Path(__file__).parent.parent

if getattr(sys, "frozen", False):
    # PyInstaller bundle: everything is on sys.path
    _PIE_ROOT = Path(sys._MEIPASS) if hasattr(sys, "_MEIPASS") else Path(sys.executable).parent
    _CORE_ROOT = _PIE_ROOT
else:
    # Development mode: sibling directories
    _PIE_ROOT = _NPC_ENGINE_ROOT.parent / "plug-in-intelligence-engine"
    _CORE_ROOT = _NPC_ENGINE_ROOT.parent / "densanon-core"

    # Add densanon-core first (canonical source)
    if _CORE_ROOT.exists() and str(_CORE_ROOT) not in sys.path:
        sys.path.insert(0, str(_CORE_ROOT))

    # Add PIE (for PluginIntelligenceEngine orchestrator)
    if _PIE_ROOT.exists() and str(_PIE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PIE_ROOT))

    if not _PIE_ROOT.exists():
        raise ImportError(
            f"PIE not found at {_PIE_ROOT}. "
            f"Expected plug-in-intelligence-engine as a sibling directory."
        )

# Re-export from densanon-core (canonical source)
from densanon.core.model_loader import BaseModel  # noqa: E402, F401
from densanon.core.expert_system import Expert, ExpertRouter, SolvedExample, ExpertResult  # noqa: E402, F401
from densanon.core.fusion import FusionLayer  # noqa: E402, F401
from densanon.core.memory import MemorySystem  # noqa: E402, F401
from densanon.core.pipeline import Pipeline  # noqa: E402, F401
from densanon.core.config import Config as PIEConfig  # noqa: E402, F401
from densanon.core.cache import SpeculativeEngine, KVCacheManager  # noqa: E402, F401
from densanon.core.router import Router  # noqa: E402, F401
from densanon.core.modules import ModuleManager  # noqa: E402, F401

# PIE's main engine class (the orchestrator — lives in PIE, not core)
from main import PluginIntelligenceEngine  # noqa: E402, F401

# Paths
PIE_ROOT = _PIE_ROOT
NPC_ENGINE_ROOT = _NPC_ENGINE_ROOT
CORE_ROOT = _CORE_ROOT
