"""
Labeled dataset for the Narrative Judge feasibility test.

144 hand-authored (fact, quest_objective, label) tuples.
4 Ashenvale quests x 3 classes x 12 examples per class.

Labels:
  advance — fact directly supports the quest's success condition
  block   — fact contradicts / negates the quest's success condition
  neutral — fact is unrelated to the quest (or only weakly related)

Drawn from Ashenvale standing tensions in
data/story_director/ashenvale_lore.md.
"""

QUESTS = {
    "missing_hammers":
        "The player identified who has been stealing tools from Kael's forge.",
    "counterfeit_steel":
        "The player exposed Mara's counterfeit steel smuggling operation.",
    "silverwood_ruins":
        "The player found evidence of elven ruins hidden in the Silverwood.",
    "wolf_bounty":
        "The player collected wolf pelts for Roderick's standing bounty.",
}


# Each entry is (fact_text, quest_id, label).
DATASET = [
    # ─── missing_hammers / advance (12) ───
    ("Mara was caught red-handed sneaking out of Kael's forge with a sack of tools.", "missing_hammers", "advance"),
    ("Roderick found Mara's distinctive footprints in the ash by Kael's anvil.", "missing_hammers", "advance"),
    ("Pip saw a hooded figure leaving Kael's forge at midnight and identified Mara from her gait.", "missing_hammers", "advance"),
    ("Mara confessed to stealing Kael's hammers to fund her counterfeit work.", "missing_hammers", "advance"),
    ("The player tracked the thief back to Mara's storeroom and found Kael's hammers hidden there.", "missing_hammers", "advance"),
    ("Bess identified Mara from the inn window the night the tools went missing.", "missing_hammers", "advance"),
    ("Roderick arrested Mara at dawn for theft from Kael's forge.", "missing_hammers", "advance"),
    ("A child found one of Kael's missing hammers stamped with his maker's mark in Mara's wagon.", "missing_hammers", "advance"),
    ("Mara's signed confession of stealing Kael's hammers is on Roderick's desk.", "missing_hammers", "advance"),
    ("Pip caught Mara in the act of pocketing Kael's smallest hammer at the forge.", "missing_hammers", "advance"),
    ("Mara admitted to the player that she took Kael's hammers because she needed the steel.", "missing_hammers", "advance"),
    ("The thief who stole Kael's hammers turned out to be Mara, witnessed by three villagers.", "missing_hammers", "advance"),

    # ─── missing_hammers / block (12) ───
    ("Kael's forge runs full tonight - no tools have gone missing all week.", "missing_hammers", "block"),
    ("Kael recounted his hammers - none are actually missing, he miscounted.", "missing_hammers", "block"),
    ("The 'stolen' hammers were found in Kael's own cellar where he'd left them.", "missing_hammers", "block"),
    ("Kael has decided to drop all accusations of theft. There was no thief.", "missing_hammers", "block"),
    ("Forensic exam shows no signs of forced entry into Kael's forge at any point.", "missing_hammers", "block"),
    ("Kael publicly apologized to Mara for falsely accusing her of theft.", "missing_hammers", "block"),
    ("The hammers were never stolen - Kael had lent them to a traveling smith.", "missing_hammers", "block"),
    ("Roderick formally closed the missing-hammers case as unfounded.", "missing_hammers", "block"),
    ("Kael burned his own ledger out of embarrassment over the false theft claim.", "missing_hammers", "block"),
    ("Witnesses confirm Kael himself moved the hammers to his back room.", "missing_hammers", "block"),
    ("Kael's apprentice returned with the 'missing' tools he'd taken to repair.", "missing_hammers", "block"),
    ("The whole village now knows the missing-hammers story was made up.", "missing_hammers", "block"),

    # ─── missing_hammers / neutral (12) ───
    ("Bess served stew to three travelers at the inn tonight.", "missing_hammers", "neutral"),
    ("Elara gathered wormwood from the eastern meadow at dawn.", "missing_hammers", "neutral"),
    ("Noah read his sealed letter again by candlelight and put it away.", "missing_hammers", "neutral"),
    ("A peddler set up a fruit stand in the square - gone by sundown.", "missing_hammers", "neutral"),
    ("Pip chased a stray cat through the marketplace.", "missing_hammers", "neutral"),
    ("Roderick polished his guard captain badge before evening rounds.", "missing_hammers", "neutral"),
    ("Bess hung fresh herbs to dry in the inn's rafters.", "missing_hammers", "neutral"),
    ("The Silverwood was unusually quiet today - no wolf howls, no birdsong.", "missing_hammers", "neutral"),
    ("Mara restocked her shop with foreign silk from the capital.", "missing_hammers", "neutral"),
    ("Pip ran errands for Bess in exchange for breakfast.", "missing_hammers", "neutral"),
    ("A traveling bard sang in the square for two hours and left.", "missing_hammers", "neutral"),
    ("Rain fell heavily through the night, flooding the western road.", "missing_hammers", "neutral"),

    # ─── counterfeit_steel / advance (12) ───
    ("Mara's crates of counterfeit steel were seized at the gate by Roderick.", "counterfeit_steel", "advance"),
    ("Pip stole a piece of steel from Mara's wagon - it bent like soft tin, proving the counterfeit.", "counterfeit_steel", "advance"),
    ("Roderick arrested Mara at the gate with three crates of counterfeit steel.", "counterfeit_steel", "advance"),
    ("Bess overheard Mara whisper 'the marks on the crates' to her courier, proving the smuggling.", "counterfeit_steel", "advance"),
    ("Mara's hidden ledger lists every counterfeit shipment for the past two years.", "counterfeit_steel", "advance"),
    ("Kael tested a merchant-guild blade and proved it was made from Mara's counterfeit steel.", "counterfeit_steel", "advance"),
    ("The player presented Roderick with stamped evidence of Mara's fake steel shipments.", "counterfeit_steel", "advance"),
    ("Mara confessed publicly to running counterfeit steel through Ashenvale for two seasons.", "counterfeit_steel", "advance"),
    ("A foreign buyer testified Mara sold him fake steel for the price of real.", "counterfeit_steel", "advance"),
    ("Mara's smuggling routes were mapped and shut down by the constabulary.", "counterfeit_steel", "advance"),
    ("The player caught Mara stamping false maker's marks on inferior steel ingots.", "counterfeit_steel", "advance"),
    ("Mara was publicly exposed as the counterfeit-steel smuggler at the harvest festival.", "counterfeit_steel", "advance"),

    # ─── counterfeit_steel / block (12) ───
    ("Mara's counterfeit shipment reached the capital and equipped two regiments.", "counterfeit_steel", "block"),
    ("Mara won her court case proving the steel accusations were unfounded.", "counterfeit_steel", "block"),
    ("Inspection found all of Mara's steel to be genuine, properly stamped, fully legal.", "counterfeit_steel", "block"),
    ("The guild certified Mara's steel as authentic dwarf-forged metal.", "counterfeit_steel", "block"),
    ("Mara expanded her steel trade with the capital under royal license.", "counterfeit_steel", "block"),
    ("Kael apologized publicly for falsely accusing Mara of counterfeiting steel.", "counterfeit_steel", "block"),
    ("Roderick withdrew all charges against Mara for lack of evidence.", "counterfeit_steel", "block"),
    ("The 'counterfeit' steel turned out to be a rare alloy nobody knew existed.", "counterfeit_steel", "block"),
    ("Mara's steel was certified by the king's own armorer as the finest grade.", "counterfeit_steel", "block"),
    ("All accusations of counterfeit steel against Mara were formally dismissed.", "counterfeit_steel", "block"),
    ("The merchants' guild gave Mara an award for steel quality this season.", "counterfeit_steel", "block"),
    ("Mara's smuggling rumors were proven to be Kael's jealous fabrication.", "counterfeit_steel", "block"),

    # ─── counterfeit_steel / neutral (12) ───
    ("Elara gathered rare ferns from a deep grove near the Silverwood.", "counterfeit_steel", "neutral"),
    ("Bess swept the inn's main hall after the breakfast rush.", "counterfeit_steel", "neutral"),
    ("Pip played dice with a pair of farmers behind the smithy.", "counterfeit_steel", "neutral"),
    ("Noah told the children a story of the old king at midwinter.", "counterfeit_steel", "neutral"),
    ("A new well was dug in the village square this week.", "counterfeit_steel", "neutral"),
    ("Roderick mended a tear in his guard's surcoat by the fire.", "counterfeit_steel", "neutral"),
    ("The chickens at the inn laid more eggs than usual this morning.", "counterfeit_steel", "neutral"),
    ("Kael forged a set of nails for the temple roof repair.", "counterfeit_steel", "neutral"),
    ("Pip and Bess walked together to the river to wash laundry.", "counterfeit_steel", "neutral"),
    ("Elara taught a young apprentice how to identify yarrow leaves.", "counterfeit_steel", "neutral"),
    ("The cooper finished a new barrel for the autumn cider press.", "counterfeit_steel", "neutral"),
    ("The miller's wife brought fresh bread to Bess at the inn.", "counterfeit_steel", "neutral"),

    # ─── silverwood_ruins / advance (12) ───
    ("Elara produced a silver medallion etched with an elven sigil and admitted she found it in the woods.", "silverwood_ruins", "advance"),
    ("Elara led the player to a moss-covered archway buried under three centuries of leaves.", "silverwood_ruins", "advance"),
    ("The player discovered a hidden elven temple in the deepest part of the Silverwood.", "silverwood_ruins", "advance"),
    ("Carved elven script runs along stone slabs found beneath the Silverwood.", "silverwood_ruins", "advance"),
    ("Elara confirmed the silver medallion was elven craftsmanship from before the kingdom.", "silverwood_ruins", "advance"),
    ("The player mapped the foundations of an ancient elven city under the Silverwood canopy.", "silverwood_ruins", "advance"),
    ("Noah confirmed the player's photos show genuine pre-kingdom elven runes.", "silverwood_ruins", "advance"),
    ("A scholar from the capital authenticated the elven artifacts the player recovered from the Silverwood.", "silverwood_ruins", "advance"),
    ("The player uncovered an elven sanctuary still intact beneath the Silverwood roots.", "silverwood_ruins", "advance"),
    ("Elara guided the player to elven gravestones never recorded on any map.", "silverwood_ruins", "advance"),
    ("The carved stone fragment Elara was rubbing dirt from is verified elven origin.", "silverwood_ruins", "advance"),
    ("The player's expedition into the Silverwood returned with proof of elven habitation.", "silverwood_ruins", "advance"),

    # ─── silverwood_ruins / block (12) ───
    ("Elara declared the Silverwood empty - only trees and wolves, nothing more.", "silverwood_ruins", "block"),
    ("Surveyors found no trace of any structure in the Silverwood, ancient or otherwise.", "silverwood_ruins", "block"),
    ("The 'elven medallion' turned out to be a recent forgery sold by a traveling peddler.", "silverwood_ruins", "block"),
    ("Noah recanted his earlier story - there are no elven ruins, that was a child's tale.", "silverwood_ruins", "block"),
    ("The carved stones were natural rock formations, not artifacts.", "silverwood_ruins", "block"),
    ("The kingdom's archaeological survey confirmed no elven presence in the Silverwood, ever.", "silverwood_ruins", "block"),
    ("The supposed elven script was identified as common dwarvish road markings.", "silverwood_ruins", "block"),
    ("Elara admits she fabricated the elven medallion to draw tourists to the village.", "silverwood_ruins", "block"),
    ("The 'temple' the player found was just an old wolf den with collapsed walls.", "silverwood_ruins", "block"),
    ("The royal cartographers have no record of any structures in the Silverwood region.", "silverwood_ruins", "block"),
    ("The elven kingdom never extended to this part of the continent, scholars confirm.", "silverwood_ruins", "block"),
    ("All claims of elven ruins in the Silverwood have been formally debunked.", "silverwood_ruins", "block"),

    # ─── silverwood_ruins / neutral (12) ───
    ("Bess prepared mutton stew for the evening meal at the inn.", "silverwood_ruins", "neutral"),
    ("Pip lost his shoes at the river and walked home barefoot.", "silverwood_ruins", "neutral"),
    ("Roderick negotiated a hay price with a farmer from the west road.", "silverwood_ruins", "neutral"),
    ("Kael sharpened twenty plough-blades for the spring planting season.", "silverwood_ruins", "neutral"),
    ("The village dog had puppies in the cooper's shed this morning.", "silverwood_ruins", "neutral"),
    ("Mara's wagon broke an axle on the way back from the capital.", "silverwood_ruins", "neutral"),
    ("A cold north wind brought the first snowfall of the year.", "silverwood_ruins", "neutral"),
    ("Elara mended a torn cloak for the orphan twins by the fire.", "silverwood_ruins", "neutral"),
    ("Noah dozed by the fire while the village settled into night.", "silverwood_ruins", "neutral"),
    ("Two travelers passed through Ashenvale without stopping, heading east.", "silverwood_ruins", "neutral"),
    ("The temple bell cracked during the noon ringing and needs repair.", "silverwood_ruins", "neutral"),
    ("Pip dreams of being a guard like Roderick when he grows up.", "silverwood_ruins", "neutral"),

    # ─── wolf_bounty / advance (12) ───
    ("Roderick paid the player five gold for three wolf pelts brought to the guardhouse.", "wolf_bounty", "advance"),
    ("Roderick weighed three wolf pelts the player brought in and paid fifteen gold.", "wolf_bounty", "advance"),
    ("The player skinned three wolves at the edge of the Silverwood and stored the pelts.", "wolf_bounty", "advance"),
    ("The player turned in a full wagonload of wolf pelts to claim the standing bounty.", "wolf_bounty", "advance"),
    ("Roderick added the player's name to the guard's wolf-bounty ledger for ten pelts.", "wolf_bounty", "advance"),
    ("The player collected the wolf bounty payment of fifty gold from the guardhouse.", "wolf_bounty", "advance"),
    ("Roderick recorded the player as the season's top wolf-pelt earner.", "wolf_bounty", "advance"),
    ("The player presented six wolf-skulls to Roderick to claim the kill confirmation.", "wolf_bounty", "advance"),
    ("Roderick stamped the player's bounty receipt for eight wolf pelts.", "wolf_bounty", "advance"),
    ("The player handed over fifteen wolf pelts and received the full standing bounty.", "wolf_bounty", "advance"),
    ("The royal courier confirmed the player's wolf-bounty claim and authorized payment.", "wolf_bounty", "advance"),
    ("The player completed the standing wolf-pelt bounty with twenty pelts delivered.", "wolf_bounty", "advance"),

    # ─── wolf_bounty / block (12) ───
    ("The royal courier announced the wolf bounty has been formally rescinded.", "wolf_bounty", "block"),
    ("The standing wolf bounty was repealed by the crown last week.", "wolf_bounty", "block"),
    ("Roderick has no funds to pay any wolf bounty this season.", "wolf_bounty", "block"),
    ("The king's tax collector seized all wolf-bounty payments before they reached the guard.", "wolf_bounty", "block"),
    ("The wolves have all migrated north - no pelts can be found in this region.", "wolf_bounty", "block"),
    ("All standing bounties in Ashenvale are suspended pending royal review.", "wolf_bounty", "block"),
    ("The wolf population has been protected by royal decree; bounty hunting is illegal.", "wolf_bounty", "block"),
    ("Roderick refused all bounty claims this season due to a budget freeze.", "wolf_bounty", "block"),
    ("The wolf-bounty proclamation was a forgery - there was never an official bounty.", "wolf_bounty", "block"),
    ("The guard treasury is empty; no wolf bounties will be paid this year.", "wolf_bounty", "block"),
    ("Wolves are now protected creatures in Ashenvale and no pelts may be turned in.", "wolf_bounty", "block"),
    ("The standing wolf bounty was struck from the kingdom's ledgers.", "wolf_bounty", "block"),

    # ─── wolf_bounty / neutral (12) ───
    ("Bess sang an old ballad about the Silverwood after closing the inn.", "wolf_bounty", "neutral"),
    ("Elara restocked her shop with fresh willowbark from the riverside.", "wolf_bounty", "neutral"),
    ("Kael fitted new horseshoes onto the miller's draught horse.", "wolf_bounty", "neutral"),
    ("Noah cleaned and oiled his old sword in his cottage.", "wolf_bounty", "neutral"),
    ("Pip helped Bess carry water from the well at sunset.", "wolf_bounty", "neutral"),
    ("A new family moved into the abandoned house at the village edge.", "wolf_bounty", "neutral"),
    ("The harvest festival is being planned for the next full moon.", "wolf_bounty", "neutral"),
    ("Mara opened her shop late and closed early three days in a row.", "wolf_bounty", "neutral"),
    ("Roderick spoke to the village children about basic guard duties.", "wolf_bounty", "neutral"),
    ("The cooper's son broke his arm falling from the apple tree.", "wolf_bounty", "neutral"),
    ("Bess hosted a music night at the inn with three local musicians.", "wolf_bounty", "neutral"),
    ("The river rose two feet after the long rainstorm last week.", "wolf_bounty", "neutral"),
]


def get_class_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, _, label in DATASET:
        counts[label] = counts.get(label, 0) + 1
    return counts


def get_quest_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, qid, _ in DATASET:
        counts[qid] = counts.get(qid, 0) + 1
    return counts


if __name__ == "__main__":
    print(f"Total tuples: {len(DATASET)}")
    print(f"Class counts: {get_class_counts()}")
    print(f"Quest counts: {get_quest_counts()}")
