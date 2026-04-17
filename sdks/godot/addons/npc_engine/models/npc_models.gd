class_name NPCModels
extends RefCounted
## Data models for NPC Engine API responses.
##
## Each inner class represents a structured response from the NPC Engine REST API.
## Use the static [code]from_dict()[/code] factory methods to parse JSON dictionaries
## returned by the server.


## Dialogue content parsed from the NPC's generated response.
class DialogueContent extends RefCounted:
	var dialogue: String = ""
	var emotion: String = "neutral"
	var action: String = ""
	var quest: QuestData = null

	static func from_dict(data: Dictionary) -> DialogueContent:
		var obj := DialogueContent.new()
		obj.dialogue = data.get("dialogue", "")
		obj.emotion = data.get("emotion", "neutral")
		obj.action = data.get("action", "")
		if data.has("quest") and data["quest"] is Dictionary:
			obj.quest = QuestData.from_dict(data["quest"])
		return obj


## Quest data embedded within dialogue content.
class QuestData extends RefCounted:
	var type: String = ""
	var objective: String = ""
	var reward: String = ""

	static func from_dict(data: Dictionary) -> QuestData:
		var obj := QuestData.new()
		obj.type = data.get("type", "")
		obj.objective = data.get("objective", "")
		obj.reward = data.get("reward", "")
		return obj


## Result of a generate (dialogue) request.
class GenerateResult extends RefCounted:
	var npc_id: String = ""
	var raw_response: String = ""
	var parsed: DialogueContent = null
	var generation_time: float = 0.0
	var capability_state: Dictionary = {}

	static func from_dict(data: Dictionary) -> GenerateResult:
		var obj := GenerateResult.new()
		obj.npc_id = data.get("npc_id", "")
		obj.generation_time = float(data.get("generation_time", 0.0))
		obj.capability_state = data.get("capability_state", {})

		## The server returns double-encoded JSON:
		## {"response": "{\"dialogue\":\"...\", \"emotion\":\"...\"}"}
		var raw: String = data.get("response", "")
		obj.raw_response = raw

		var inner := JSON.parse_string(raw)
		if inner is Dictionary:
			obj.parsed = DialogueContent.from_dict(inner)
		else:
			## Fallback: treat the raw string as plain dialogue text.
			var fallback := DialogueContent.new()
			fallback.dialogue = raw
			obj.parsed = fallback

		return obj


## Information about a single NPC.
class NPCInfo extends RefCounted:
	var id: String = ""
	var name: String = ""
	var role: String = ""
	var capabilities: PackedStringArray = PackedStringArray()

	static func from_dict(data: Dictionary) -> NPCInfo:
		var obj := NPCInfo.new()
		obj.id = data.get("id", data.get("npc_id", ""))
		obj.name = data.get("name", "")
		obj.role = data.get("role", "")
		var caps: Array = data.get("capabilities", [])
		for cap in caps:
			obj.capabilities.append(str(cap))
		return obj


## Result of listing all NPCs in the world.
class NPCListResult extends RefCounted:
	var active: String = ""
	var world_name: String = ""
	var npcs: Array[NPCInfo] = []

	static func from_dict(data: Dictionary) -> NPCListResult:
		var obj := NPCListResult.new()
		obj.active = data.get("active", data.get("active_npc", ""))
		obj.world_name = data.get("world_name", data.get("world", ""))
		var npc_list: Array = data.get("npcs", [])
		for npc_data in npc_list:
			if npc_data is Dictionary:
				obj.npcs.append(NPCInfo.from_dict(npc_data))
		return obj


## Result of adjusting trust between the player and an NPC.
class TrustResult extends RefCounted:
	var npc_id: String = ""
	var old_level: int = 0
	var new_level: int = 0
	var delta: int = 0
	var reason: String = ""

	static func from_dict(data: Dictionary) -> TrustResult:
		var obj := TrustResult.new()
		obj.npc_id = data.get("npc_id", "")
		obj.old_level = int(data.get("old_level", data.get("old_trust", 0)))
		obj.new_level = int(data.get("new_level", data.get("new_trust", 0)))
		obj.delta = int(data.get("delta", 0))
		obj.reason = data.get("reason", "")
		return obj


## Result of setting an NPC's mood.
class MoodResult extends RefCounted:
	var npc_id: String = ""
	var old_mood: String = ""
	var new_mood: String = ""
	var intensity: float = 0.0

	static func from_dict(data: Dictionary) -> MoodResult:
		var obj := MoodResult.new()
		obj.npc_id = data.get("npc_id", "")
		obj.old_mood = data.get("old_mood", "")
		obj.new_mood = data.get("new_mood", data.get("mood", ""))
		obj.intensity = float(data.get("intensity", 0.0))
		return obj


## Result of injecting a world event.
class EventResult extends RefCounted:
	var injected: bool = false
	var target: String = ""
	var event_description: String = ""

	static func from_dict(data: Dictionary) -> EventResult:
		var obj := EventResult.new()
		obj.injected = data.get("injected", data.get("success", false))
		obj.target = data.get("target", data.get("npc_id", ""))
		obj.event_description = data.get("event_description", data.get("description", ""))
		return obj


## Result of a health check against the server.
class HealthResult extends RefCounted:
	var status: String = ""
	var version: String = ""

	static func from_dict(data: Dictionary) -> HealthResult:
		var obj := HealthResult.new()
		obj.status = data.get("status", "")
		obj.version = data.get("version", "")
		return obj


# ---------------------------------------------------------------------------
# Story Director models
# ---------------------------------------------------------------------------

## Result of a story reset.
class StoryResetResult extends RefCounted:
	var ok: bool = false
	var born_removed: PackedStringArray = PackedStringArray()
	var deceased_restored: PackedStringArray = PackedStringArray()
	var manifest_reloaded: PackedStringArray = PackedStringArray()

	static func from_dict(data: Dictionary) -> StoryResetResult:
		var obj := StoryResetResult.new()
		obj.ok = data.get("ok", false)
		for item in data.get("born_removed", []):
			obj.born_removed.append(str(item))
		for item in data.get("deceased_restored", []):
			obj.deceased_restored.append(str(item))
		for item in data.get("manifest_reloaded", []):
			obj.manifest_reloaded.append(str(item))
		return obj


## Result of setting or getting player activity.
class ActivityResult extends RefCounted:
	var ok: bool = false
	var activity: String = ""
	var activity_set_at_tick: int = 0

	static func from_dict(data: Dictionary) -> ActivityResult:
		var obj := ActivityResult.new()
		obj.ok = data.get("ok", false)
		obj.activity = data.get("activity", "")
		obj.activity_set_at_tick = int(data.get("activity_set_at_tick", 0))
		return obj


## Result of pausing or resuming the story.
class PauseResult extends RefCounted:
	var ok: bool = false
	var paused: bool = false
	var paused_at_tick: int = 0

	static func from_dict(data: Dictionary) -> PauseResult:
		var obj := PauseResult.new()
		obj.ok = data.get("ok", false)
		obj.paused = data.get("paused", false)
		obj.paused_at_tick = int(data.get("paused_at_tick", 0))
		return obj


## Result of querying the pause state with budget info.
class PauseStateResult extends RefCounted:
	var paused: bool = false
	var paused_at_tick: int = 0
	var tick_budget_seconds: float = -1.0
	var window_llm_seconds_used: float = 0.0
	var budget_exceeded: bool = false
	var next_tick_recommended_in_seconds: float = 0.0

	static func from_dict(data: Dictionary) -> PauseStateResult:
		var obj := PauseStateResult.new()
		obj.paused = data.get("paused", false)
		obj.paused_at_tick = int(data.get("paused_at_tick", 0))
		obj.tick_budget_seconds = float(data.get("tick_budget_seconds", -1.0))
		obj.window_llm_seconds_used = float(data.get("window_llm_seconds_used", 0.0))
		obj.budget_exceeded = data.get("budget_exceeded", false)
		obj.next_tick_recommended_in_seconds = float(data.get("next_tick_recommended_in_seconds", 0.0))
		return obj


## Result of setting the tick budget.
class TickBudgetResult extends RefCounted:
	var ok: bool = false
	var tick_budget_seconds: float = -1.0

	static func from_dict(data: Dictionary) -> TickBudgetResult:
		var obj := TickBudgetResult.new()
		obj.ok = data.get("ok", false)
		obj.tick_budget_seconds = float(data.get("tick_budget_seconds", -1.0))
		return obj


## Result of setting or getting quest pacing.
class QuestPacingResult extends RefCounted:
	var ok: bool = false
	var max_unoffered: int = 0
	var cooldown_ticks: int = 0

	static func from_dict(data: Dictionary) -> QuestPacingResult:
		var obj := QuestPacingResult.new()
		obj.ok = data.get("ok", false)
		obj.max_unoffered = int(data.get("max_unoffered", 0))
		obj.cooldown_ticks = int(data.get("cooldown_ticks", 0))
		return obj


## Result of refusing a quest.
class QuestRefusalResult extends RefCounted:
	var ok: bool = false
	var quest_id: String = ""
	var npc_id: String = ""
	var trust_delta: int = 0
	var refusal_mode: String = ""

	static func from_dict(data: Dictionary) -> QuestRefusalResult:
		var obj := QuestRefusalResult.new()
		obj.ok = data.get("ok", false)
		obj.quest_id = data.get("quest_id", "")
		obj.npc_id = data.get("npc_id", "")
		obj.trust_delta = int(data.get("trust_delta", 0))
		obj.refusal_mode = data.get("refusal_mode", "")
		return obj


## Result of setting or getting auto-refuse config.
class AutoRefuseResult extends RefCounted:
	var ok: bool = false
	var intents: PackedStringArray = PackedStringArray()
	var dev_enabled: bool = false

	static func from_dict(data: Dictionary) -> AutoRefuseResult:
		var obj := AutoRefuseResult.new()
		obj.ok = data.get("ok", false)
		for item in data.get("intents", []):
			obj.intents.append(str(item))
		obj.dev_enabled = data.get("dev_enabled", false)
		return obj


## Result of introducing the player to an NPC.
class IntroduceResult extends RefCounted:
	var ok: bool = false
	var npc_id: String = ""
	var known_as: PackedStringArray = PackedStringArray()
	var max_trust: int = 0

	static func from_dict(data: Dictionary) -> IntroduceResult:
		var obj := IntroduceResult.new()
		obj.ok = data.get("ok", false)
		obj.npc_id = data.get("npc_id", "")
		for item in data.get("known_as", []):
			obj.known_as.append(str(item))
		obj.max_trust = int(data.get("max_trust", 0))
		return obj


## Result of setting a player visible feature.
class VisibleFeatureResult extends RefCounted:
	var ok: bool = false
	var player_visible_feature: String = ""

	static func from_dict(data: Dictionary) -> VisibleFeatureResult:
		var obj := VisibleFeatureResult.new()
		obj.ok = data.get("ok", false)
		obj.player_visible_feature = data.get("player_visible_feature", "")
		return obj


## Result of registering a feature-identity mapping.
class RegisterFeatureResult extends RefCounted:
	var ok: bool = false
	var feature: String = ""
	var identity: String = ""

	static func from_dict(data: Dictionary) -> RegisterFeatureResult:
		var obj := RegisterFeatureResult.new()
		obj.ok = data.get("ok", false)
		obj.feature = data.get("feature", "")
		obj.identity = data.get("identity", "")
		return obj


## Result of one NPC vouching the player to another.
class VouchResult extends RefCounted:
	var ok: bool = false
	var voucher_npc: String = ""
	var to_npc: String = ""
	var known_as: PackedStringArray = PackedStringArray()
	var max_trust: int = 0

	static func from_dict(data: Dictionary) -> VouchResult:
		var obj := VouchResult.new()
		obj.ok = data.get("ok", false)
		obj.voucher_npc = data.get("voucher_npc", "")
		obj.to_npc = data.get("to_npc", "")
		for item in data.get("known_as", []):
			obj.known_as.append(str(item))
		obj.max_trust = int(data.get("max_trust", 0))
		return obj
