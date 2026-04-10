#!/usr/bin/env python3
"""
NPC Benchmark v2 — Comparison report.

Loads all available npc_v2_*.json result files and prints:
  - Side-by-side leaderboards (baseline / variant_a / variant_b)
  - Per-model lift table (dscore, dqps)
  - Per-dimension breakdown (which variant fixes which weakness)
  - Final recommendation (best model + best variant by quality and by Q/sec)
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent

# Files to load (skip ones that don't exist)
SOURCES = [
    ("baseline",        "npc_v2_baseline.json"),
    ("baseline_med",    "npc_v2_baseline_medium.json"),
    ("variant_a",       "npc_v2_variant_a.json"),
    ("variant_b",       "npc_v2_variant_b.json"),
]

DIMS = ["identity", "knowledge", "events", "quests", "valid_json",
        "hallucination_grace", "contradiction_recovery", "ood_deflection"]
DIM_SHORT = ["id", "kn", "ev", "qu", "js", "hl", "co", "od"]


def load_runs():
    """Returns dict[run_tag -> dict[model_name -> result]]."""
    runs = {}
    for tag, fname in SOURCES:
        path = ROOT / fname
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        runs[tag] = {r["model"]: r for r in data["results"]}
    return runs


def main():
    runs = load_runs()
    if not runs:
        print("No npc_v2_*.json files found")
        return

    # Merge baseline + baseline_med under one logical "baseline"
    baseline = {}
    baseline.update(runs.get("baseline", {}))
    baseline.update(runs.get("baseline_med", {}))
    runs["baseline_full"] = baseline

    print("=" * 90)
    print("  NPC BENCHMARK v2 - COMPARISON REPORT")
    print("=" * 90)

    # ── 1. Combined baseline leaderboard ──
    print(f"\n[1] BASELINE LEADERBOARD (no PIE module, all tested models)")
    print(f"    {'Model':<16s} {'Size':>6s} {'Score':>9s} {'sec/call':>9s} {'tok/s':>7s} {'Q/sec':>7s}")
    print(f"    {'-'*16} {'-'*6} {'-'*9} {'-'*9} {'-'*7} {'-'*7}")
    for r in sorted(baseline.values(), key=lambda x: x["total_score"], reverse=True):
        print(f"    {r['model']:<16s} {r['size_mb']:>4d}MB "
              f"{r['total_score']:>4d}/{r['max_score']:<3d} "
              f"{r['avg_time_per_call']:>9.2f} "
              f"{r['tok_per_sec']:>7.1f} "
              f"{r['quality_per_sec']:>7.3f}")

    # ── 2. Variant lift table ──
    if "variant_a" in runs or "variant_b" in runs:
        print(f"\n[2] VARIANT LIFT (vs baseline)")
        print(f"    Models tested with variants: {sorted(runs.get('variant_a', {}).keys() or runs.get('variant_b', {}).keys())}")
        print()
        header = f"    {'Model':<16s} {'Base':>5s} {'VarA':>5s} {'dA':>5s} {'VarB':>5s} {'dB':>5s} | {'B-Q/s':>6s} {'A-Q/s':>6s} {'V-A-Q/s':>8s} {'V-B-Q/s':>8s}"
        print(header)
        print(f"    {'-'*16} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*5} | {'-'*6} {'-'*6} {'-'*8} {'-'*8}")
        models = sorted(set(baseline.keys())
                        | set(runs.get("variant_a", {}).keys())
                        | set(runs.get("variant_b", {}).keys()))
        for m in models:
            b = baseline.get(m)
            va = runs.get("variant_a", {}).get(m)
            vb = runs.get("variant_b", {}).get(m)
            if not b:
                continue
            base_s = b["total_score"]
            base_qs = b["quality_per_sec"]
            va_s = va["total_score"] if va else None
            vb_s = vb["total_score"] if vb else None
            va_qs = va["quality_per_sec"] if va else None
            vb_qs = vb["quality_per_sec"] if vb else None
            delta_a = (va_s - base_s) if va_s is not None else None
            delta_b = (vb_s - base_s) if vb_s is not None else None
            print(f"    {m:<16s} {base_s:>5d} "
                  f"{(str(va_s) if va_s is not None else '-'):>5s} "
                  f"{(f'{delta_a:+d}' if delta_a is not None else '-'):>5s} "
                  f"{(str(vb_s) if vb_s is not None else '-'):>5s} "
                  f"{(f'{delta_b:+d}' if delta_b is not None else '-'):>5s} | "
                  f"{base_qs:>6.3f} "
                  f"{(f'{va_qs:.3f}' if va_qs is not None else '-'):>6s} "
                  f"{(f'{va_qs - base_qs:+.3f}' if va_qs is not None else '-'):>8s} "
                  f"{(f'{vb_qs - base_qs:+.3f}' if vb_qs is not None else '-'):>8s}")

    # ── 3. Per-dimension comparison for each variant-tested model ──
    if "variant_a" in runs or "variant_b" in runs:
        print(f"\n[3] PER-DIMENSION LIFT (out of 7 NPCs each)")
        models_v = sorted(set(runs.get("variant_a", {}).keys())
                          | set(runs.get("variant_b", {}).keys()))
        for m in models_v:
            b = baseline.get(m)
            va = runs.get("variant_a", {}).get(m)
            vb = runs.get("variant_b", {}).get(m)
            if not b:
                continue
            print(f"\n    {m}:")
            print(f"      {'dim':<10s} {'base':>6s} {'varA':>6s} {'dA':>5s} {'varB':>6s} {'dB':>5s}")
            print(f"      {'-'*10} {'-'*6} {'-'*6} {'-'*5} {'-'*6} {'-'*5}")
            for dim in DIMS:
                bs = b["scores"][dim]
                vas = va["scores"][dim] if va else None
                vbs = vb["scores"][dim] if vb else None
                da = (vas - bs) if vas is not None else None
                db = (vbs - bs) if vbs is not None else None
                print(f"      {dim:<10s} {bs:>4d}/7 "
                      f"{(f'{vas}/7' if vas is not None else '-'):>6s} "
                      f"{(f'{da:+d}' if da is not None else '-'):>5s} "
                      f"{(f'{vbs}/7' if vbs is not None else '-'):>6s} "
                      f"{(f'{db:+d}' if db is not None else '-'):>5s}")

    # ── 4. Final recommendation ──
    print(f"\n[4] RECOMMENDATION")
    # Best by quality across all (model, variant) combinations
    candidates = []
    for tag, by_model in runs.items():
        if tag in ("baseline_med", "baseline_full"):
            continue
        for r in by_model.values():
            candidates.append((tag, r))
    if candidates:
        best_q = max(candidates, key=lambda x: x[1]["total_score"])
        best_qs = max(candidates, key=lambda x: x[1]["quality_per_sec"])
        print(f"    Best quality:    {best_q[1]['model']} + {best_q[0]} "
              f"=> {best_q[1]['total_score']}/{best_q[1]['max_score']} "
              f"({best_q[1]['quality_per_sec']:.3f} Q/sec)")
        print(f"    Best efficiency: {best_qs[1]['model']} + {best_qs[0]} "
              f"=> {best_qs[1]['total_score']}/{best_qs[1]['max_score']} "
              f"({best_qs[1]['quality_per_sec']:.3f} Q/sec)")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
