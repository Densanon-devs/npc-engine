"""
Narrative Judge — feasibility prototype.

Adapts GE-Sim 2.0's World Judge architecture (arXiv 2605.27491) to the
Story Director. The vision-LM + MLP head trained on labeled robot
success frames doesn't transfer cheaply, but the SCORING SHAPE does:
given (current_state, task_instruction) -> success probability.

This prototype tests whether the existing NLI cross-encoder (already
loaded by ContradictionChecker for contradiction detection) can play
that role for narrative state -> quest objective scoring, with zero
additional training.

Run:
    python narrative_judge_prototype.py

Two sections:
  Part A: 12 hand-crafted triples (fact, quest_objective, expected_dir)
          with expected_dir in {advance, block, neutral}. Tests whether
          NLI's three-way output cleanly separates the three categories.
  Part B: 50-tick replay over a synthesized FactLedger drawn from the
          Ashenvale lore + standing tensions. Each entry scored against
          4 active quests. Reports best-match distributions and per-tick
          score time-series.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Optional


NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
LABELS = ("contradiction", "entailment", "neutral")


def load_nli():
    """Lazy-load the same NLI cross-encoder ContradictionChecker uses."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        print("ERROR: sentence-transformers not installed.")
        sys.exit(1)
    print(f"Loading NLI model: {NLI_MODEL_NAME} (first run downloads ~140MB)...")
    model = CrossEncoder(NLI_MODEL_NAME)
    print("NLI model loaded.\n")
    return model


def score(model, premise: str, hypothesis: str) -> dict:
    """Return softmaxed label probabilities + the picked label."""
    raw = model.predict([(premise, hypothesis)])[0]
    scores = [float(s) for s in raw]
    mx = max(scores)
    exps = [math.exp(s - mx) for s in scores]
    total = sum(exps)
    probs = [e / total for e in exps]
    label_idx = max(range(len(probs)), key=lambda i: probs[i])
    return {
        "label": LABELS[label_idx],
        "confidence": probs[label_idx],
        "scores": {LABELS[i]: probs[i] for i in range(len(LABELS))},
    }


# ─────────────────────────────────────────────────────────────────
# Part A — hand-crafted feasibility triples
# ─────────────────────────────────────────────────────────────────

# Drawn from Ashenvale lore + examples.yaml. The premise is the FactLedger
# entry text the Director recorded; the hypothesis is a quest objective
# stating what success looks like. Expected direction is what the
# narrative *should* see: does this fact advance the quest, block it,
# or is it unrelated?

TRIPLES = [
    # Quest: Find out who stole Kael's hammers
    {
        "premise": "Mara was caught sneaking out of Kael's forge with a sack of tools.",
        "hypothesis": "The player identified the person stealing from Kael's forge.",
        "expected": "advance",
        "note": "Direct evidence — caught in the act.",
    },
    {
        "premise": "Kael's forge was undisturbed all week. No tools missing.",
        "hypothesis": "The player identified the person stealing from Kael's forge.",
        "expected": "block",
        "note": "Negates the premise of the quest.",
    },
    {
        "premise": "Bess served stew to three travelers at the inn tonight.",
        "hypothesis": "The player identified the person stealing from Kael's forge.",
        "expected": "neutral",
        "note": "Unrelated village activity.",
    },

    # Quest: Stop the counterfeit steel from reaching market
    {
        "premise": "Mara's crates of counterfeit steel were seized at the gate.",
        "hypothesis": "The counterfeit steel was prevented from reaching the market.",
        "expected": "advance",
        "note": "Direct success state.",
    },
    {
        "premise": "Mara sold the entire counterfeit shipment to merchants from the capital.",
        "hypothesis": "The counterfeit steel was prevented from reaching the market.",
        "expected": "block",
        "note": "Direct failure state.",
    },
    {
        "premise": "Pip the urchin watched the sunset from the rooftop.",
        "hypothesis": "The counterfeit steel was prevented from reaching the market.",
        "expected": "neutral",
        "note": "Unrelated trivial event.",
    },

    # Quest: Investigate the Silverwood elven ruins
    {
        "premise": "Elara returned from the Silverwood carrying a carved elven medallion.",
        "hypothesis": "The player discovered evidence of elven ruins in the Silverwood.",
        "expected": "advance",
        "note": "Concrete artifact evidence.",
    },
    {
        "premise": "Elara reports the Silverwood is empty — only trees and wolves, nothing more.",
        "hypothesis": "The player discovered evidence of elven ruins in the Silverwood.",
        "expected": "block",
        "note": "Explicit negation by domain expert.",
    },
    {
        "premise": "Kael repaired the broken hinge on the inn's back door.",
        "hypothesis": "The player discovered evidence of elven ruins in the Silverwood.",
        "expected": "neutral",
        "note": "Unrelated chore.",
    },

    # Quest: Collect wolf pelts for the standing bounty
    {
        "premise": "Roderick paid the player five gold for three wolf pelts brought to the guardhouse.",
        "hypothesis": "The player turned in wolf pelts for the standing bounty.",
        "expected": "advance",
        "note": "Direct payment event.",
    },
    {
        "premise": "The wolf bounty was rescinded by royal decree this morning.",
        "hypothesis": "The player turned in wolf pelts for the standing bounty.",
        "expected": "block",
        "note": "Quest precondition removed.",
    },
    {
        "premise": "Bess sang a tavern song about the Silverwood after closing.",
        "hypothesis": "The player turned in wolf pelts for the standing bounty.",
        "expected": "neutral",
        "note": "Unrelated atmosphere.",
    },
]


def run_part_a(model) -> dict:
    """Score each triple, return confusion matrix + accuracy."""
    print("=" * 72)
    print("PART A — feasibility triples (12 hand-crafted)")
    print("=" * 72)
    print()

    # NLI labels map naturally to our expected directions:
    #   entailment    -> advance  (state implies success condition)
    #   contradiction -> block    (state contradicts success condition)
    #   neutral       -> neutral
    label_to_direction = {
        "entailment": "advance",
        "contradiction": "block",
        "neutral": "neutral",
    }

    confusion = {"advance": {}, "block": {}, "neutral": {}}
    for exp in confusion:
        confusion[exp] = {"advance": 0, "block": 0, "neutral": 0}

    rows = []
    correct = 0
    for i, t in enumerate(TRIPLES, 1):
        result = score(model, t["premise"], t["hypothesis"])
        predicted = label_to_direction[result["label"]]
        match = "OK" if predicted == t["expected"] else "MISS"
        if predicted == t["expected"]:
            correct += 1
        confusion[t["expected"]][predicted] += 1
        rows.append({
            "i": i,
            "expected": t["expected"],
            "predicted": predicted,
            "confidence": result["confidence"],
            "scores": result["scores"],
            "match": match,
            "premise": t["premise"][:60] + ("..." if len(t["premise"]) > 60 else ""),
            "hypothesis": t["hypothesis"][:60] + ("..." if len(t["hypothesis"]) > 60 else ""),
            "note": t["note"],
        })

    # Print individual results
    print(f"{'#':>3} {'exp':>8} {'pred':>8} {'conf':>5} {'ent':>5} {'con':>5} {'neu':>5}  premise / hypothesis")
    print("-" * 100)
    for r in rows:
        s = r["scores"]
        print(f"{r['i']:>3} {r['expected']:>8} {r['predicted']:>8} {r['confidence']:>5.2f} "
              f"{s['entailment']:>5.2f} {s['contradiction']:>5.2f} {s['neutral']:>5.2f}  "
              f"[{r['match']}] {r['premise']}")
        print(f"{'':>56}    -> {r['hypothesis']}")

    print()
    print("Confusion (rows = expected, cols = predicted):")
    print(f"{'':>10} {'advance':>8} {'block':>8} {'neutral':>8}")
    for exp in ("advance", "block", "neutral"):
        c = confusion[exp]
        print(f"{exp:>10} {c['advance']:>8} {c['block']:>8} {c['neutral']:>8}")
    print()
    print(f"Accuracy: {correct}/{len(TRIPLES)} = {100*correct/len(TRIPLES):.1f}%")
    print()
    return {
        "confusion": confusion,
        "accuracy": correct / len(TRIPLES),
        "rows": rows,
    }


# ─────────────────────────────────────────────────────────────────
# Part B — 50-tick replay
# ─────────────────────────────────────────────────────────────────

# 50 FactLedger entries synthesized from Ashenvale standing tensions +
# world facts. Mix:
#   - Direct advances for one of 4 active quests
#   - Direct contradictions for one quest
#   - Neutral village color
#   - Tangentially-related (NPC mentioned but action unrelated)

ACTIVE_QUESTS = [
    {
        "id": "missing_hammers",
        "objective": "The player identified who has been stealing tools from Kael's forge.",
    },
    {
        "id": "counterfeit_steel",
        "objective": "The player exposed Mara's counterfeit steel smuggling operation.",
    },
    {
        "id": "silverwood_ruins",
        "objective": "The player found evidence of elven ruins hidden in the Silverwood.",
    },
    {
        "id": "wolf_bounty",
        "objective": "The player collected wolf pelts for Roderick's standing bounty.",
    },
]

# 50 synthesized FactLedger entries. Authored to test: do scores cleanly
# track which quest a tick advances, and do neutral entries score low
# across all quests?
REPLAY_ENTRIES = [
    # Hammers thread (T1-T8)
    ("Kael discovered another hammer missing from the forge this morning.", "hammers-context"),
    ("Pip saw a hooded figure leaving Kael's forge at midnight.", "hammers-clue"),
    ("Mara denied being near the forge when Kael confronted her.", "hammers-dodge"),
    ("Roderick found Mara's footprints in the ash by Kael's anvil.", "hammers-evidence"),
    ("Mara confessed to stealing Kael's hammers to fund her counterfeit work.", "hammers-resolution"),
    ("Kael's forge runs full tonight — no tools have gone missing all week.", "hammers-contradiction"),
    ("Bess served stew to three travelers at the inn tonight.", "neutral-color"),
    ("Pip chased a stray cat through the marketplace.", "neutral-color"),

    # Counterfeit steel thread (T9-T18)
    ("Mara received a crate of unmarked steel from a Northern Port courier.", "steel-context"),
    ("Kael tested a merchant-guild blade and the steel rang false.", "steel-clue"),
    ("Roderick noted Mara's wagons leave Ashenvale heavy and return empty.", "steel-clue"),
    ("Bess overheard Mara whisper about 'the marks on the crates' to her courier.", "steel-clue"),
    ("Pip stole a piece of steel from Mara's wagon — it bent like soft tin.", "steel-evidence"),
    ("Roderick arrested Mara at the gate with three crates of counterfeit steel.", "steel-resolution"),
    ("Mara's counterfeit shipment reached the capital and equipped two regiments.", "steel-contradiction"),
    ("Elara gathered wormwood from the eastern meadow at dawn.", "neutral-color"),
    ("Noah read his sealed letter again by candlelight and put it away.", "neutral-color"),
    ("A peddler set up a fruit stand in the square — gone by sundown.", "neutral-color"),

    # Silverwood ruins thread (T19-T28)
    ("Elara returned from the Silverwood unusually quiet today.", "silverwood-context"),
    ("Elara was seen rubbing dirt from a carved stone fragment in her shop.", "silverwood-clue"),
    ("Pip swears he saw lights moving through the Silverwood canopy at night.", "silverwood-clue"),
    ("Noah remembered his grandfather mentioning 'old stones' deeper than the wolf trails.", "silverwood-clue"),
    ("Elara produced a silver medallion etched with an elven sigil and admitted she found it in the woods.", "silverwood-evidence"),
    ("Elara led the player to a moss-covered archway buried under three centuries of leaves.", "silverwood-resolution"),
    ("Elara declared the Silverwood empty — only trees and wolves, nothing more.", "silverwood-contradiction"),
    ("Roderick polished his guard captain badge before evening rounds.", "neutral-color"),
    ("Bess hung fresh herbs to dry in the inn's rafters.", "neutral-color"),
    ("Kael repaired the broken hinge on the cooper's back door.", "neutral-color"),

    # Wolf bounty thread (T29-T38)
    ("A farmer reported wolves taking sheep from his northern pasture.", "wolf-context"),
    ("Roderick reminded the village that the wolf pelt bounty still stands at five gold per pelt.", "wolf-context"),
    ("Pip saw a wolf carcass dragged behind a hunter's wagon at sundown.", "wolf-clue"),
    ("The player skinned three wolves at the edge of the Silverwood.", "wolf-progress"),
    ("Roderick weighed three wolf pelts the player brought in and paid fifteen gold.", "wolf-resolution"),
    ("The royal courier announced the wolf bounty has been formally rescinded.", "wolf-contradiction"),
    ("Mara restocked her shop with foreign silk from the capital.", "neutral-color"),
    ("Noah dozed by the fire while the village settled into night.", "neutral-color"),
    ("Bess sang an old ballad about the Silverwood after closing.", "neutral-color"),
    ("Pip ran errands for Bess in exchange for breakfast.", "neutral-color"),

    # Cross-thread mixes (T39-T50) — verify clean separation
    ("Kael accused Mara of being behind both the missing hammers AND the counterfeit steel.", "hammers+steel-mix"),
    ("Elara mentioned seeing wolf tracks near the elven archway in the Silverwood.", "wolf+silverwood-mix"),
    ("Mara claimed her steel comes from a Silverwood quarry no one has ever mapped.", "steel+silverwood-mix"),
    ("Roderick suspects Pip stole the hammers, but he has no proof yet.", "hammers-misdirection"),
    ("Noah told the player to leave the Silverwood alone — some things should stay buried.", "silverwood-warning"),
    ("Elara warned that the wolves grow bolder near the buried elven stones.", "wolf+silverwood-mix"),
    ("The tax collector still has not returned from his last visit to Ashenvale.", "background-lore"),
    ("Bess remembered every traveler who passed through the inn this season.", "background-lore"),
    ("Roderick drinks alone in the guardhouse on the nights Mara's wagons run.", "steel-color"),
    ("Pip dreams of being a guard like Roderick when he grows up.", "neutral-color"),
    ("Kael forged a new set of nails for the temple roof repair.", "neutral-color"),
    ("The Silverwood was unusually quiet today — no wolf howls, no birdsong.", "silverwood-color"),
]


def run_part_b(model) -> dict:
    """Score each replay entry against all 4 active quests."""
    print("=" * 72)
    print(f"PART B — 50-tick replay ({len(REPLAY_ENTRIES)} entries x {len(ACTIVE_QUESTS)} quests)")
    print("=" * 72)
    print()

    per_tick = []
    for tick, (text, tag) in enumerate(REPLAY_ENTRIES, 1):
        quest_scores = {}
        for q in ACTIVE_QUESTS:
            r = score(model, text, q["objective"])
            quest_scores[q["id"]] = r
        # Best-entailment quest = the quest this tick most ADVANCES
        best_quest_id = max(
            quest_scores.keys(),
            key=lambda qid: quest_scores[qid]["scores"]["entailment"],
        )
        best_ent = quest_scores[best_quest_id]["scores"]["entailment"]
        per_tick.append({
            "tick": tick,
            "tag": tag,
            "text": text,
            "best_quest": best_quest_id,
            "best_entailment": best_ent,
            "scores": {qid: quest_scores[qid]["scores"] for qid in quest_scores},
        })

    # Tick-by-tick best-quest assignment
    print(f"{'tick':>4} {'tag':>22} {'best-quest':>20} {'ent':>5}  text")
    print("-" * 130)
    for t in per_tick:
        print(f"{t['tick']:>4} {t['tag']:>22} {t['best_quest']:>20} {t['best_entailment']:>5.2f}  {t['text'][:75]}")
    print()

    # Aggregate: did each quest "fire" (high-entailment tick) only on its
    # designated subset?
    print("Per-quest analysis:")
    print()
    EXPECT_PREFIX = {
        "missing_hammers": "hammers",
        "counterfeit_steel": "steel",
        "silverwood_ruins": "silverwood",
        "wolf_bounty": "wolf",
    }
    for q in ACTIVE_QUESTS:
        qid = q["id"]
        expect_prefix = EXPECT_PREFIX[qid]
        # Entailment scores for this quest across all ticks
        ents = [(t["tick"], t["tag"], t["text"], t["scores"][qid]["entailment"],
                 t["scores"][qid]["contradiction"], t["scores"][qid]["neutral"])
                for t in per_tick]
        ents.sort(key=lambda x: -x[3])
        print(f"Quest [{qid}]:  {q['objective']}")
        print(f"  Top 5 entailment ticks:")
        for tick, tag, text, ent, con, neu in ents[:5]:
            on_topic = "ON " if expect_prefix in tag else "off"
            print(f"    T{tick:>2} ent={ent:.2f} con={con:.2f} neu={neu:.2f} [{on_topic}] {tag:>22}  {text[:65]}")
        # Contradiction-detection check: does the explicit contradiction
        # entry for this quest actually score high contradiction?
        contradiction_tag = f"{expect_prefix}-contradiction"
        contradiction_ticks = [t for t in per_tick if t["tag"] == contradiction_tag]
        if contradiction_ticks:
            for ct in contradiction_ticks:
                s = ct["scores"][qid]
                ranked_label = max(s.keys(), key=lambda k: s[k])
                hit = "HIT" if ranked_label == "contradiction" else "MISS"
                print(f"  Contradiction probe: [{hit}] T{ct['tick']} ranked={ranked_label} "
                      f"con={s['contradiction']:.2f} ent={s['entailment']:.2f} neu={s['neutral']:.2f}")
                print(f"     text: {ct['text']}")
        # Off-topic floor: what's the highest entailment from an off-topic tick?
        off_topic_max = max(
            (e[3] for e in ents if expect_prefix not in e[1]),
            default=0.0,
        )
        # On-topic floor: lowest entailment from an on-topic non-contradiction tick
        on_topic_ents = [e[3] for e in ents
                         if expect_prefix in e[1] and "contradiction" not in e[1]]
        on_topic_min = min(on_topic_ents) if on_topic_ents else 0.0
        on_topic_max = max(on_topic_ents) if on_topic_ents else 0.0
        gap = on_topic_min - off_topic_max
        print(f"  Separability: on-topic range [{on_topic_min:.2f}, {on_topic_max:.2f}]  "
              f"vs off-topic max {off_topic_max:.2f}  -> gap={gap:+.2f}")
        print()

    # Score distribution histogram
    all_best_ents = [t["best_entailment"] for t in per_tick]
    buckets = [0, 0, 0, 0, 0]  # 0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0
    for e in all_best_ents:
        i = min(int(e * 5), 4)
        buckets[i] += 1
    print("Best-quest entailment distribution across all 50 ticks:")
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    for lab, count in zip(labels, buckets):
        bar = "#" * count
        print(f"  {lab}: {count:>2}  {bar}")
    print()

    return {"per_tick": per_tick}


def main():
    model = load_nli()
    part_a = run_part_a(model)
    print()
    part_b = run_part_b(model)

    # Final verdict
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    print()
    print(f"Part A accuracy: {100*part_a['accuracy']:.1f}%  ({sum(part_a['confusion'][k][k] for k in part_a['confusion'])}/{len(TRIPLES)})")
    print()
    print("Part B: see per-quest separability gaps above.")
    print("  - gap > 0.20:  Narrative Judge is viable")
    print("  - gap 0.05-0.20: marginal — threshold tuning may help")
    print("  - gap < 0.05:  NLI cannot distinguish quest-relevance cleanly")
    print()


if __name__ == "__main__":
    main()
