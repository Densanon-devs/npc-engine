#!/usr/bin/env python3
"""Final 10-model NPC benchmark leaderboard.

Merges:
  - npc_v2_baseline.json     (10-model run, but 3B models had broken expert path)
  - npc_v2_baseline_3b_fixed.json  (3B re-run after _should_enable_experts fix)

The 3B models from the fixed run override the broken results in the original.
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent

baseline = json.loads((ROOT / "npc_v2_baseline.json").read_text(encoding="utf-8"))
fixed_3b = json.loads((ROOT / "npc_v2_baseline_3b_fixed.json").read_text(encoding="utf-8"))

# Build a name->result map starting from the original baseline
results = {r["model"]: r for r in baseline["results"]}
# Override the 3 broken 3B models with the fixed run
for r in fixed_3b["results"]:
    results[r["model"]] = r

# Sort by total_score desc
all_results = sorted(results.values(), key=lambda r: r["total_score"], reverse=True)

print("=" * 96)
print("  NPC BENCHMARK v2 - FINAL 10-MODEL LEADERBOARD (after 3B expert-path fix)")
print("=" * 96)

# ── 1. Quality leaderboard ──
print(f"\n[1] LEADERBOARD - sorted by total quality")
print(f"    {'Model':<22s} {'Size':>7s} {'Score':>9s} {'sec/call':>9s} {'tok/s':>7s} {'Q/sec':>7s}")
print(f"    {'-'*22} {'-'*7} {'-'*9} {'-'*9} {'-'*7} {'-'*7}")
for r in all_results:
    print(f"    {r['model']:<22s} {r['size_mb']:>4d}MB "
          f"{r['total_score']:>4d}/{r['max_score']:<3d} "
          f"{r['avg_time_per_call']:>9.2f} "
          f"{r['tok_per_sec']:>7.1f} "
          f"{r['quality_per_sec']:>7.3f}")

# ── 2. Efficiency leaderboard ──
print(f"\n[2] LEADERBOARD - sorted by quality/sec (efficiency)")
print(f"    {'Model':<22s} {'Size':>7s} {'Q/sec':>7s} {'Score':>9s} {'sec/call':>9s}")
print(f"    {'-'*22} {'-'*7} {'-'*7} {'-'*9} {'-'*9}")
for r in sorted(all_results, key=lambda r: r["quality_per_sec"], reverse=True):
    print(f"    {r['model']:<22s} {r['size_mb']:>4d}MB "
          f"{r['quality_per_sec']:>7.3f} "
          f"{r['total_score']:>4d}/{r['max_score']:<3d} "
          f"{r['avg_time_per_call']:>9.2f}")

# ── 3. Per-dimension breakdown ──
DIMS = ["identity", "knowledge", "events", "quests", "valid_json",
        "hallucination_grace", "contradiction_recovery", "ood_deflection"]
print(f"\n[3] PER-DIMENSION BREAKDOWN (out of 7 NPCs each)")
header = f"    {'Model':<22s} " + " ".join(f"{d[:5]:>5s}" for d in DIMS)
print(header)
print("    " + "-" * (22 + 6 * len(DIMS)))
for r in all_results:
    cells = " ".join(f"{r['scores'][d]:>3d}/7" for d in DIMS)
    print(f"    {r['model']:<22s} {cells}")

# ── 4. Per-NPC breakdown for the top 3 ──
print(f"\n[4] PER-NPC PASS RATES for top 3 models")
top3 = all_results[:3]
NPCS = ["noah", "kael", "mara", "guard_roderick", "elara", "bess", "pip"]
print(f"    {'Model':<22s} " + " ".join(f"{n[:6]:>7s}" for n in NPCS))
print(f"    {'-'*22} " + " ".join("-" * 7 for _ in NPCS))
for r in top3:
    if "per_npc" not in r:
        continue
    cells = []
    for npc in NPCS:
        n = r["per_npc"].get(npc, {})
        passes = sum(1 for k in DIMS if n.get(k))
        cells.append(f"{passes:>3d}/8")
    print(f"    {r['model']:<22s} " + "    ".join(cells))

# ── 5. Recommendation ──
print(f"\n[5] RECOMMENDATION")
print(f"    Best quality:    {all_results[0]['model']} ({all_results[0]['size_mb']}MB) "
      f"= {all_results[0]['total_score']}/{all_results[0]['max_score']} "
      f"({all_results[0]['quality_per_sec']:.3f} Q/sec)")
best_qs = max(all_results, key=lambda r: r["quality_per_sec"])
print(f"    Best efficiency: {best_qs['model']} ({best_qs['size_mb']}MB) "
      f"= {best_qs['total_score']}/{best_qs['max_score']} "
      f"({best_qs['quality_per_sec']:.3f} Q/sec)")

# Save merged result
merged = {
    "tag": "baseline_final",
    "timestamp": baseline["timestamp"],
    "max_score_per_model": baseline["max_score_per_model"],
    "results": all_results,
}
(ROOT / "npc_v2_baseline_final.json").write_text(
    json.dumps(merged, indent=2), encoding="utf-8")

print(f"\n  Merged JSON saved to npc_v2_baseline_final.json")
print("=" * 96)
