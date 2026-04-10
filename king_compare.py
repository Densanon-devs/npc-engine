#!/usr/bin/env python3
"""Compare baseline vs targeted variants for the quality king (Llama 3.2 3B)
and the speed king (Qwen2.5 0.5B). Per-dimension lifts and regressions."""

import json
from pathlib import Path

ROOT = Path(__file__).parent
DIMS = ["identity", "knowledge", "events", "quests", "valid_json",
        "hallucination_grace", "contradiction_recovery", "ood_deflection"]
DIM_LABEL = ["id", "kn", "ev", "qu", "js", "hl", "co", "od"]


def load(name):
    p = ROOT / f"npc_v2_{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def get_model(data, model_name):
    if not data:
        return None
    for r in data["results"]:
        if r["model"] == model_name:
            return r
    return None


def diff_row(label, base, alt):
    if alt is None:
        return f"  {label:<20s} {'(missing)':>10s}"
    score_diff = alt["total_score"] - base["total_score"]
    qs_diff = alt["quality_per_sec"] - base["quality_per_sec"]
    sign = "+" if score_diff >= 0 else ""
    qs_sign = "+" if qs_diff >= 0 else ""
    return (f"  {label:<20s} "
            f"{alt['total_score']:>3d}/56  ({sign}{score_diff:+d})  "
            f"{alt['quality_per_sec']:>5.3f} ({qs_sign}{qs_diff:+.3f})  "
            f"sec/call={alt['avg_time_per_call']:.2f}")


def dim_table(label, base, runs):
    print(f"\n  {label} per-dimension:")
    print(f"    {'dim':<6s} {'base':>5s}", end="")
    for run_label, _ in runs:
        print(f" {run_label:>10s}", end="")
    print()
    for dim, dim_label in zip(DIMS, DIM_LABEL):
        bs = base["scores"][dim]
        print(f"    {dim_label:<6s} {bs:>3d}/7", end="")
        for _, run_data in runs:
            if run_data is None:
                print(f" {'-':>10s}", end="")
                continue
            s = run_data["scores"][dim]
            d = s - bs
            arrow = ("+" if d > 0 else " ") if d >= 0 else ""
            print(f" {s:>2d}/7({arrow}{d:+d})", end="")
        print()


def main():
    print("=" * 90)
    print("  TARGETED VARIANTS — QUALITY KING vs SPEED KING")
    print("=" * 90)

    # ── Quality king: Llama 3.2 3B ──
    base_q = get_model(load("baseline_3b_fixed"), "Llama 3.2 3B")
    llama_runs = [
        ("varA", get_model(load("king_llama3b_varA"), "Llama 3.2 3B")),
        ("varB", get_model(load("king_llama3b_varB"), "Llama 3.2 3B")),
        ("temp 0.5", get_model(load("king_llama3b_temp05"), "Llama 3.2 3B")),
    ]
    print(f"\n[Quality king] Llama 3.2 3B (baseline = {base_q['total_score']}/56, "
          f"{base_q['quality_per_sec']:.3f} Q/sec, {base_q['avg_time_per_call']:.2f}s/call)")
    print(f"  {'-'*78}")
    print(f"  baseline             {base_q['total_score']:>3d}/56  (------)  "
          f"{base_q['quality_per_sec']:>5.3f} (-------)  sec/call={base_q['avg_time_per_call']:.2f}")
    for label, run in llama_runs:
        print(diff_row(label, base_q, run))
    dim_table("Llama 3.2 3B", base_q, llama_runs)

    # ── Speed king: Qwen2.5 0.5B ──
    base_s = get_model(load("baseline"), "Qwen2.5 0.5B")
    qwen_runs = [
        ("varA", get_model(load("king_qwen05b_varA"), "Qwen2.5 0.5B")),
        ("varB", get_model(load("king_qwen05b_varB"), "Qwen2.5 0.5B")),
        ("temp 0.3", get_model(load("king_qwen05b_temp03"), "Qwen2.5 0.5B")),
    ]
    print(f"\n[Speed king] Qwen2.5 0.5B (baseline = {base_s['total_score']}/56, "
          f"{base_s['quality_per_sec']:.3f} Q/sec, {base_s['avg_time_per_call']:.2f}s/call)")
    print(f"  {'-'*78}")
    print(f"  baseline             {base_s['total_score']:>3d}/56  (------)  "
          f"{base_s['quality_per_sec']:>5.3f} (-------)  sec/call={base_s['avg_time_per_call']:.2f}")
    for label, run in qwen_runs:
        print(diff_row(label, base_s, run))
    dim_table("Qwen2.5 0.5B", base_s, qwen_runs)

    # ── Final picks ──
    print(f"\n[Recommendations]")
    llama_best = max(
        [("baseline", base_q)] + llama_runs,
        key=lambda x: x[1]["total_score"] if x[1] else -1,
    )
    qwen_best = max(
        [("baseline", base_s)] + qwen_runs,
        key=lambda x: x[1]["total_score"] if x[1] else -1,
    )
    print(f"  Llama 3.2 3B winner: {llama_best[0]} = {llama_best[1]['total_score']}/56")
    print(f"  Qwen2.5 0.5B winner: {qwen_best[0]} = {qwen_best[1]['total_score']}/56 "
          f"({qwen_best[1]['quality_per_sec']:.3f} Q/sec)")
    print("=" * 90)


if __name__ == "__main__":
    main()
