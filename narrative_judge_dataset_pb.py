"""
Port Blackwater labeled dataset for cross-domain Narrative Judge validation.

72 hand-authored (fact, quest_objective, label) tuples.
4 PB quests x 3 classes x 6 examples per class.

Drawn from Port Blackwater standing tensions in
data/worlds/port_blackwater/story/lore.md (pirate port, NOT Ashenvale).

Same label semantics as narrative_judge_dataset.py:
  advance / block / neutral.
"""

QUESTS_PB = {
    "dark_lighthouse":
        "The player discovered what caused the north-cliff lighthouse to go dark and found the missing keeper.",
    "smuggling_tunnels":
        "The player exposed Old Bones' smuggling operation through the tunnels under the Drowned Rat tavern.",
    "finn_glass_shard":
        "The player recovered the melted lighthouse-lens shard Finn hid at the end of dock three.",
    "shoals_mystery":
        "The player investigated what is causing ships to wreck on the Shoals and identified the source of the midnight singing.",
}


DATASET_PB = [
    # ─── dark_lighthouse / advance (6) ───
    ("The player found the lighthouse keeper imprisoned in a sea cave near the cliff.", "dark_lighthouse", "advance"),
    ("Old Bones revealed that the keeper was kidnapped by the Red Tide fleet two months ago.", "dark_lighthouse", "advance"),
    ("The player located the lighthouse keeper alive in the smuggling tunnels.", "dark_lighthouse", "advance"),
    ("Finn produced the keeper's logbook, which named her captors before the light went out.", "dark_lighthouse", "advance"),
    ("The player identified the Red Tide captain as the one who darkened the lighthouse for smuggling cover.", "dark_lighthouse", "advance"),
    ("The investigation concluded: the lighthouse keeper was paid off by the Red Tide to abandon her post.", "dark_lighthouse", "advance"),

    # ─── dark_lighthouse / block (6) ───
    ("The lighthouse keeper had simply quit and moved to Meridia for personal reasons unrelated to the dark light.", "dark_lighthouse", "block"),
    ("The lighthouse went dark due to a broken lens - no foul play, no missing keeper.", "dark_lighthouse", "block"),
    ("The keeper was found dead from natural causes weeks before the light went out.", "dark_lighthouse", "block"),
    ("Captain Reva confirmed the lighthouse was decommissioned by official decree months ago.", "dark_lighthouse", "block"),
    ("The 'mystery' of the dark lighthouse was solved as a routine maintenance failure.", "dark_lighthouse", "block"),
    ("Investigation closed: the lighthouse keeper retired peacefully; there is no mystery.", "dark_lighthouse", "block"),

    # ─── dark_lighthouse / neutral (6) ───
    ("Old Bones served grog to a half-dozen sailors at the Drowned Rat tonight.", "dark_lighthouse", "neutral"),
    ("Finn coiled rope at the end of dock four after his shift.", "dark_lighthouse", "neutral"),
    ("Captain Reva inspected the new mooring lines for the eastern berths.", "dark_lighthouse", "neutral"),
    ("A gull stole a fish from a sailor's basket on the quay.", "dark_lighthouse", "neutral"),
    ("Old Bones played dice with three off-duty sailors by the tavern fire.", "dark_lighthouse", "neutral"),
    ("Finn fell asleep on a coil of rope by sundown.", "dark_lighthouse", "neutral"),

    # ─── smuggling_tunnels / advance (6) ───
    ("Captain Reva caught Old Bones moving sealed crates through the tunnel beneath the Drowned Rat at midnight.", "smuggling_tunnels", "advance"),
    ("The player mapped the smuggling tunnels from the tavern cellar to the cave system under the cliff.", "smuggling_tunnels", "advance"),
    ("Old Bones confessed to running contraband through the tunnels for the past five years.", "smuggling_tunnels", "advance"),
    ("Finn led Reva to the hidden tunnel entrance behind the tavern's wine racks.", "smuggling_tunnels", "advance"),
    ("The player produced a shipment ledger proving Old Bones routed black-market goods through Drowned Rat tunnels.", "smuggling_tunnels", "advance"),
    ("Reva arrested Old Bones with two crates of contraband from the tunnel mouth.", "smuggling_tunnels", "advance"),

    # ─── smuggling_tunnels / block (6) ───
    ("Captain Reva inspected the cellar of the Drowned Rat and found no tunnel - only barrels and dust.", "smuggling_tunnels", "block"),
    ("Old Bones produced clean trade licenses for every cargo Reva had suspected him of smuggling.", "smuggling_tunnels", "block"),
    ("The supposed tunnels under the Drowned Rat turned out to be solid bedrock, no passage at all.", "smuggling_tunnels", "block"),
    ("Reva publicly apologized to Old Bones for the false smuggling accusations after the inspection cleared him.", "smuggling_tunnels", "block"),
    ("All charges of smuggling against Old Bones were formally dropped for lack of evidence.", "smuggling_tunnels", "block"),
    ("The Drowned Rat passed three surprise audits this season - no contraband, no tunnels.", "smuggling_tunnels", "block"),

    # ─── smuggling_tunnels / neutral (6) ───
    ("Finn stacked barrels on dock two for the dawn shipment to Meridia.", "smuggling_tunnels", "neutral"),
    ("Captain Reva trimmed the mainsail of her cutter before the morning patrol.", "smuggling_tunnels", "neutral"),
    ("A pelican fished off the end of dock three for an hour.", "smuggling_tunnels", "neutral"),
    ("The harbor master's bell rang the noon hour over a still port.", "smuggling_tunnels", "neutral"),
    ("Finn shared a hot meal with two dockhands at sundown.", "smuggling_tunnels", "neutral"),
    ("Old Bones tuned his fiddle by the tavern fire while a kettle whistled.", "smuggling_tunnels", "neutral"),

    # ─── finn_glass_shard / advance (6) ───
    ("Finn produced the melted lighthouse-lens shard from under the loose plank at dock three and gave it to the player.", "finn_glass_shard", "advance"),
    ("The player recovered the glowing glass shard Finn hid under the dock-three plank.", "finn_glass_shard", "advance"),
    ("Finn confessed to the player about the shard and led them to its hiding spot.", "finn_glass_shard", "advance"),
    ("The lighthouse lens shard was authenticated by Old Bones as the same glass from the keeper's tower.", "finn_glass_shard", "advance"),
    ("The player found the melted shard at the end of dock three exactly where Finn had hidden it.", "finn_glass_shard", "advance"),
    ("Finn handed the player the sea-tongue-marked glass shard he'd been guarding for weeks.", "finn_glass_shard", "advance"),

    # ─── finn_glass_shard / block (6) ───
    ("Finn lost the glass shard during a storm - it washed out to sea and cannot be recovered.", "finn_glass_shard", "block"),
    ("The shard Finn claimed to find turned out to be a piece of common bottle-green glass, not lens material.", "finn_glass_shard", "block"),
    ("Finn admitted he made up the story of the glowing glass shard to seem important.", "finn_glass_shard", "block"),
    ("The plank at the end of dock three rotted through and the shard fell into the harbor unrecoverable.", "finn_glass_shard", "block"),
    ("Investigation found no evidence any glass shard ever existed - Finn's story was a dream.", "finn_glass_shard", "block"),
    ("The glass shard was destroyed in the tavern fire last week before anyone could examine it.", "finn_glass_shard", "block"),

    # ─── finn_glass_shard / neutral (6) ───
    ("Captain Reva drafted the week's harbor patrol schedule by lamplight.", "finn_glass_shard", "neutral"),
    ("Old Bones swept the tavern floor before opening for the evening crowd.", "finn_glass_shard", "neutral"),
    ("Finn ate a bowl of fish stew at the Drowned Rat after his shift.", "finn_glass_shard", "neutral"),
    ("Reva polished the brass fittings on her cutter's wheel.", "finn_glass_shard", "neutral"),
    ("A traveling merchant unloaded silk at dock one for the morning market.", "finn_glass_shard", "neutral"),
    ("Old Bones told a long story about a kraken to a crowd of three sailors.", "finn_glass_shard", "neutral"),

    # ─── shoals_mystery / advance (6) ───
    ("The player discovered an ancient sunken temple beneath the Shoals - its bells chime in tidal currents, which is the 'singing' Old Bones heard.", "shoals_mystery", "advance"),
    ("The player identified the cause of the Shoals wrecks: a magnetic ore deposit on the seabed pulls ships off course.", "shoals_mystery", "advance"),
    ("Reva confirmed the singing comes from sea-tongue glyphs glowing under the waterline at midnight.", "shoals_mystery", "advance"),
    ("The player descended to the Shoals at low tide and found pre-kingdom altars beneath the rocks.", "shoals_mystery", "advance"),
    ("The investigation concluded: an ancient sunken city beneath the Shoals is the source of the midnight singing.", "shoals_mystery", "advance"),
    ("Old Bones revealed that his great-grandfather mapped the sunken city beneath the Shoals; the player has the chart.", "shoals_mystery", "advance"),

    # ─── shoals_mystery / block (6) ───
    ("Surveyors confirmed the Shoals are ordinary rocks - no sunken structures, no magnetic anomalies, no singing.", "shoals_mystery", "block"),
    ("The 'singing' Old Bones heard was identified as wind through the cliff hollows - a known acoustic effect.", "shoals_mystery", "block"),
    ("Reva commissioned a thorough survey of the Shoals and found nothing beneath but sand and bedrock.", "shoals_mystery", "block"),
    ("The Shoals mystery was debunked as a series of unrelated navigation errors by inexperienced crews.", "shoals_mystery", "block"),
    ("The kingdom's marine survey concluded the Shoals are ordinary, with no supernatural cause for the wrecks.", "shoals_mystery", "block"),
    ("All claims of singing beneath the Shoals were attributed to weather and exhaustion by the inquiry.", "shoals_mystery", "block"),

    # ─── shoals_mystery / neutral (6) ───
    ("Finn whistled while coiling rope on dock two at midday.", "shoals_mystery", "neutral"),
    ("Old Bones tapped a fresh barrel of ale before the evening crowd arrived.", "shoals_mystery", "neutral"),
    ("Captain Reva drafted her weekly report by lantern-light in her cabin.", "shoals_mystery", "neutral"),
    ("A flock of gulls wheeled over the eastern berths chasing a fishing boat.", "shoals_mystery", "neutral"),
    ("Old Bones served three plates of stew to a crew just in from the open sea.", "shoals_mystery", "neutral"),
    ("Finn watched the lamps come on along the quay at sundown.", "shoals_mystery", "neutral"),
]


def get_class_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, _, label in DATASET_PB:
        counts[label] = counts.get(label, 0) + 1
    return counts


def get_quest_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, qid, _ in DATASET_PB:
        counts[qid] = counts.get(qid, 0) + 1
    return counts


if __name__ == "__main__":
    print(f"Total tuples: {len(DATASET_PB)}")
    print(f"Class counts: {get_class_counts()}")
    print(f"Quest counts: {get_quest_counts()}")
