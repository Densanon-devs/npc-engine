#!/usr/bin/env python3
"""Download the recommended GGUF model for NPC Engine.

Downloads to plug-in-intelligence-engine/models/ (sibling directory).

Usage:
    python download_model.py              # downloads Qwen2.5 0.5B (speed king, 469MB)
    python download_model.py --quality    # downloads Llama 3.2 3B (quality king, 2GB)
    python download_model.py --both       # downloads both
"""

import argparse
import os
import sys
from pathlib import Path

MODELS = {
    "speed": {
        "name": "Qwen2.5 0.5B (speed king — 56/56 with postgen, 3s/call)",
        "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "file": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "size_mb": 469,
    },
    "quality": {
        "name": "Llama 3.2 3B (quality king — 56/56 raw, 11s/call)",
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "size_mb": 2000,
    },
}


def download_model(key: str, models_dir: Path):
    model = MODELS[key]
    dest = models_dir / model["file"]

    if dest.exists():
        print(f"  Already exists: {dest}")
        return

    print(f"  Downloading: {model['name']}")
    print(f"  From: huggingface.co/{model['repo']}")
    print(f"  Size: ~{model['size_mb']}MB")
    print(f"  To: {dest}")
    print()

    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id=model["repo"],
            filename=model["file"],
            local_dir=str(models_dir),
            local_dir_use_symlinks=False,
        )
        print(f"  Done: {dest}")
    except ImportError:
        print("  huggingface_hub not installed. Install with:")
        print("    pip install huggingface_hub")
        print()
        print("  Or download manually from:")
        print(f"    https://huggingface.co/{model['repo']}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download GGUF models for NPC Engine")
    parser.add_argument("--quality", action="store_true", help="Download Llama 3.2 3B (2GB)")
    parser.add_argument("--both", action="store_true", help="Download both models")
    args = parser.parse_args()

    # Find models directory
    npc_root = Path(__file__).parent.resolve()
    pie_root = npc_root.parent / "plug-in-intelligence-engine"
    models_dir = pie_root / "models"

    if not pie_root.exists():
        print(f"Error: PIE not found at {pie_root}")
        print("NPC Engine expects plug-in-intelligence-engine as a sibling directory.")
        sys.exit(1)

    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  NPC Engine — Model Downloader")
    print("=" * 60)

    if args.both:
        download_model("speed", models_dir)
        print()
        download_model("quality", models_dir)
    elif args.quality:
        download_model("quality", models_dir)
    else:
        download_model("speed", models_dir)

    print()
    print("  Models ready. Start the engine with:")
    print("    python -m npc_engine.server")
    print("=" * 60)


if __name__ == "__main__":
    main()
