class_name NPCEngineClient
extends Node
## HTTP client wrapper for the NPC Engine REST API.
##
## This is the main autoload singleton registered as "NPCEngine". It provides
## both fire-and-forget methods (that emit signals) and async alternatives
## (that return results via [code]await[/code]).
##
## [b]Signal-based usage:[/b]
## [codeblock]
## func _ready():
##     NPCEngine.npc_response_received.connect(_on_npc_response)
##     NPCEngine.generate("Hello there!", "blacksmith")
##
## func _on_npc_response(result: NPCModels.GenerateResult):
##     print(result.parsed.dialogue)
## [/codeblock]
##
## [b]Await-based usage:[/b]
## [codeblock]
## func _ready():
##     var result := await NPCEngine.generate_async("Hello!", "blacksmith")
##     print(result.parsed.dialogue)
## [/codeblock]

## Emitted when a generate response is received from the server.
signal npc_response_received(result: NPCModels.GenerateResult)

## Emitted when the NPC list is received.
signal npc_list_received(result: NPCModels.NPCListResult)

## Emitted when the active NPC is switched.
signal npc_switched(info: NPCModels.NPCInfo)

## Emitted when trust is adjusted for an NPC.
signal trust_adjusted(result: NPCModels.TrustResult)

## Emitted when an NPC's mood is set.
signal mood_set(result: NPCModels.MoodResult)

## Emitted when a world event is injected.
signal event_injected(result: NPCModels.EventResult)

## Emitted when a story reset completes.
signal story_reset_completed(result: NPCModels.StoryResetResult)

## Emitted when the player activity is set or retrieved.
signal activity_received(result: NPCModels.ActivityResult)

## Emitted when the story is paused or resumed.
signal pause_changed(result: NPCModels.PauseResult)

## Emitted when the pause state is retrieved.
signal pause_state_received(result: NPCModels.PauseStateResult)

## Emitted when the tick budget is set.
signal tick_budget_set(result: NPCModels.TickBudgetResult)

## Emitted when quest pacing is set or retrieved.
signal quest_pacing_received(result: NPCModels.QuestPacingResult)

## Emitted when an NPC death is queued.
signal npc_death_queued(result: Dictionary)

## Emitted when the graveyard is retrieved.
signal graveyard_received(result: Dictionary)

## Emitted when an NPC birth is queued.
signal npc_birth_queued(result: Dictionary)

## Emitted when the population is retrieved.
signal population_received(result: Dictionary)

## Emitted when a quest is refused.
signal quest_refused(result: NPCModels.QuestRefusalResult)

## Emitted when auto-refuse config is set or retrieved.
signal auto_refuse_received(result: NPCModels.AutoRefuseResult)

## Emitted when the player introduces themselves to an NPC.
signal player_introduced(result: NPCModels.IntroduceResult)

## Emitted when a visible feature is set.
signal visible_feature_set(result: NPCModels.VisibleFeatureResult)

## Emitted when a feature-identity mapping is registered.
signal feature_registered(result: NPCModels.RegisterFeatureResult)

## Emitted when an NPC vouches the player to another.
signal player_vouched(result: NPCModels.VouchResult)

## Emitted when the identity state is retrieved.
signal identity_state_received(result: Dictionary)

## Emitted when the reputation is retrieved.
signal reputation_received(result: Dictionary)

## Emitted when any API request fails.
signal request_failed(endpoint: String, error: String)

## Emitted when a health check succeeds.
signal server_connected

## Emitted when the server becomes unreachable.
signal server_disconnected

## Base URL of the NPC Engine server (no trailing slash).
@export var server_url: String = "http://127.0.0.1:8000"

## Request timeout in seconds.
@export var request_timeout: float = 30.0

## Tracks consecutive connection failures for disconnect detection.
var _consecutive_failures: int = 0

## Whether we consider the server connected.
var _connected: bool = false


# ---------------------------------------------------------------------------
# Fire-and-forget API methods (emit signals on completion)
# ---------------------------------------------------------------------------

## Generate NPC dialogue for the given prompt.
## If [param npc_id] is empty, the server uses the currently active NPC.
func generate(prompt: String, npc_id: String = "") -> void:
	var body := {"prompt": prompt}
	if npc_id != "":
		body["npc_id"] = npc_id
	_post("/generate", body, _on_generate_response)


## Request the list of all NPCs in the current world.
func list_npcs() -> void:
	_get("/npc/list", _on_list_npcs_response)


## Switch the active NPC to [param npc_id].
func switch_npc(npc_id: String) -> void:
	_post("/npc/switch", {"npc_id": npc_id}, _on_switch_npc_response)


## Inject a world event. If [param npc_id] is empty, it targets all NPCs.
func inject_event(description: String, npc_id: String = "") -> void:
	var body: Dictionary = {"description": description}
	if npc_id != "":
		body["npc_id"] = npc_id
	_post("/events/inject", body, _on_inject_event_response)


## Adjust trust between the player and an NPC.
func adjust_trust(npc_id: String, delta: int, reason: String = "") -> void:
	var body: Dictionary = {"npc_id": npc_id, "delta": delta}
	if reason != "":
		body["reason"] = reason
	_post("/npc/trust", body, _on_trust_response)


## Set an NPC's mood.
func set_mood(npc_id: String, mood: String, intensity: float = 0.5, pin_turns: int = 3) -> void:
	_post("/npc/mood", {"npc_id": npc_id, "mood": mood, "intensity": intensity, "pin_turns": pin_turns}, _on_mood_response)


## Add a scratchpad entry for an NPC (short-term memory).
func add_scratchpad(npc_id: String, text: String, importance: float = 0.7) -> void:
	_post("/npc/scratchpad", {"npc_id": npc_id, "text": text, "importance": importance}, func(_d: Dictionary) -> void:
		pass  ## No specific signal for scratchpad; request_failed will fire on error.
	)


## Accept a quest on behalf of the player.
func accept_quest(quest_id: String, quest_name: String, given_by: String) -> void:
	_post("/quests/accept", {"quest_id": quest_id, "quest_name": quest_name, "given_by": given_by}, func(_d: Dictionary) -> void:
		pass
	)


## Mark a quest as completed.
func complete_quest(quest_id: String) -> void:
	_post("/quests/complete", {"quest_id": quest_id}, func(_d: Dictionary) -> void:
		pass
	)


## Check the server health endpoint.
func check_health() -> void:
	_get("/health", _on_health_response)


# ---------------------------------------------------------------------------
# Story Director — fire-and-forget API methods
# ---------------------------------------------------------------------------

## Reset the story to YAML baseline.
func reset_story() -> void:
	_post("/story/reset", {}, _on_story_reset_response)


## Set the player's current activity context.
func set_activity(activity: String) -> void:
	_post("/story/activity", {"activity": activity}, _on_activity_response)


## Pause the story (hold all future ticks).
func pause_story() -> void:
	_post("/story/pause", {}, _on_pause_changed_response)


## Resume the story after a pause.
func resume_story() -> void:
	_post("/story/resume", {}, _on_pause_changed_response)


## Set the rolling-window LLM-time cap.
func set_tick_budget(max_seconds_per_minute: float) -> void:
	_post("/story/tick_budget", {"max_seconds_per_minute": max_seconds_per_minute}, _on_tick_budget_response)


## Override per-NPC quest pacing caps.
func set_quest_pacing(max_unoffered: int, cooldown_ticks: int) -> void:
	_post("/story/quest_pacing", {"max_unoffered": max_unoffered, "cooldown_ticks": cooldown_ticks}, _on_quest_pacing_response)


## Queue a death dispatch for an NPC (game-authoritative).
func queue_npc_death(npc_id: String, cause: String, transfers_quests_to: String = "") -> void:
	var body: Dictionary = {"npc_id": npc_id, "cause": cause}
	if transfers_quests_to != "":
		body["transfers_quests_to"] = transfers_quests_to
	_post("/story/npc_death", body, _on_npc_death_response)


## Queue a birth request for a zone.
func queue_npc_birth(zone: String, role: String = "") -> void:
	var body: Dictionary = {"zone": zone}
	if role != "":
		body["role"] = role
	_post("/story/npc_birth_request", body, _on_npc_birth_response)


## Refuse a quest on behalf of the player.
func refuse_quest(quest_id: String, npc_id: String, reason: String = "") -> void:
	var body: Dictionary = {"quest_id": quest_id, "npc_id": npc_id}
	if reason != "":
		body["reason"] = reason
	_post("/quests/refuse", body, _on_quest_refused_response)


## Set the player's auto-refuse intent filter.
func set_auto_refuse(intents: PackedStringArray) -> void:
	var intents_array: Array = []
	for intent in intents:
		intents_array.append(intent)
	_post("/player/auto_refuse", {"intents": intents_array}, _on_auto_refuse_response)


## Introduce the player to an NPC.
func introduce_player(to_npc: String, player_name: String, titles: PackedStringArray = PackedStringArray()) -> void:
	var body: Dictionary = {"to_npc": to_npc, "name": player_name}
	if titles.size() > 0:
		var titles_array: Array = []
		for title in titles:
			titles_array.append(title)
		body["titles"] = titles_array
	_post("/player/introduce", body, _on_player_introduced_response)


## Set a player-visible feature (cloak, weapon, etc.).
func set_visible_feature(feature: String) -> void:
	_post("/player/visible_feature", {"feature": feature}, _on_visible_feature_response)


## Map a visible feature to an identity for auto-recognition.
func register_feature(feature: String, identity: String) -> void:
	_post("/player/register_feature", {"feature": feature, "identity": identity}, _on_feature_registered_response)


## One NPC vouches the player to another.
func vouch_player(voucher_npc: String, to_npc: String) -> void:
	_post("/player/vouched_by", {"voucher_npc": voucher_npc, "to_npc": to_npc}, _on_player_vouched_response)


# ---------------------------------------------------------------------------
# Await-style async alternatives
# ---------------------------------------------------------------------------

## Generate NPC dialogue and return the result (use with [code]await[/code]).
func generate_async(prompt: String, npc_id: String = "") -> NPCModels.GenerateResult:
	var body := {"prompt": prompt}
	if npc_id != "":
		body["npc_id"] = npc_id

	var data := await _post_async("/generate", body)
	if data.is_empty():
		return null

	var result := NPCModels.GenerateResult.from_dict(data)
	npc_response_received.emit(result)
	return result


## List NPCs and return the result (use with [code]await[/code]).
func list_npcs_async() -> NPCModels.NPCListResult:
	var data := await _get_async("/npc/list")
	if data.is_empty():
		return null

	var result := NPCModels.NPCListResult.from_dict(data)
	npc_list_received.emit(result)
	return result


## Check health and return the result (use with [code]await[/code]).
func check_health_async() -> NPCModels.HealthResult:
	var data := await _get_async("/health")
	if data.is_empty():
		return null

	return NPCModels.HealthResult.from_dict(data)


# ---------------------------------------------------------------------------
# Story Director — await-style async alternatives
# ---------------------------------------------------------------------------

## Reset the story and return the result (use with [code]await[/code]).
func reset_story_async() -> NPCModels.StoryResetResult:
	var data := await _post_async("/story/reset", {})
	if data.is_empty():
		return null

	var result := NPCModels.StoryResetResult.from_dict(data)
	story_reset_completed.emit(result)
	return result


## Set the player activity and return the result (use with [code]await[/code]).
func set_activity_async(activity: String) -> NPCModels.ActivityResult:
	var data := await _post_async("/story/activity", {"activity": activity})
	if data.is_empty():
		return null

	var result := NPCModels.ActivityResult.from_dict(data)
	activity_received.emit(result)
	return result


## Get the current player activity (use with [code]await[/code]).
func get_activity_async() -> NPCModels.ActivityResult:
	var data := await _get_async("/story/activity")
	if data.is_empty():
		return null

	var result := NPCModels.ActivityResult.from_dict(data)
	activity_received.emit(result)
	return result


## Pause the story and return the result (use with [code]await[/code]).
func pause_story_async() -> NPCModels.PauseResult:
	var data := await _post_async("/story/pause", {})
	if data.is_empty():
		return null

	var result := NPCModels.PauseResult.from_dict(data)
	pause_changed.emit(result)
	return result


## Resume the story and return the result (use with [code]await[/code]).
func resume_story_async() -> NPCModels.PauseResult:
	var data := await _post_async("/story/resume", {})
	if data.is_empty():
		return null

	var result := NPCModels.PauseResult.from_dict(data)
	pause_changed.emit(result)
	return result


## Get the current pause state (use with [code]await[/code]).
func get_pause_state_async() -> NPCModels.PauseStateResult:
	var data := await _get_async("/story/pause_state")
	if data.is_empty():
		return null

	var result := NPCModels.PauseStateResult.from_dict(data)
	pause_state_received.emit(result)
	return result


## Set the tick budget and return the result (use with [code]await[/code]).
func set_tick_budget_async(max_seconds_per_minute: float) -> NPCModels.TickBudgetResult:
	var data := await _post_async("/story/tick_budget", {"max_seconds_per_minute": max_seconds_per_minute})
	if data.is_empty():
		return null

	var result := NPCModels.TickBudgetResult.from_dict(data)
	tick_budget_set.emit(result)
	return result


## Set quest pacing and return the result (use with [code]await[/code]).
func set_quest_pacing_async(max_unoffered: int, cooldown_ticks: int) -> NPCModels.QuestPacingResult:
	var data := await _post_async("/story/quest_pacing", {"max_unoffered": max_unoffered, "cooldown_ticks": cooldown_ticks})
	if data.is_empty():
		return null

	var result := NPCModels.QuestPacingResult.from_dict(data)
	quest_pacing_received.emit(result)
	return result


## Get current quest pacing config (use with [code]await[/code]).
func get_quest_pacing_async() -> NPCModels.QuestPacingResult:
	var data := await _get_async("/story/quest_pacing")
	if data.is_empty():
		return null

	var result := NPCModels.QuestPacingResult.from_dict(data)
	quest_pacing_received.emit(result)
	return result


## Queue an NPC death and return the raw result (use with [code]await[/code]).
func queue_npc_death_async(npc_id: String, cause: String, transfers_quests_to: String = "") -> Dictionary:
	var body: Dictionary = {"npc_id": npc_id, "cause": cause}
	if transfers_quests_to != "":
		body["transfers_quests_to"] = transfers_quests_to

	var data := await _post_async("/story/npc_death", body)
	if data.is_empty():
		return {}

	npc_death_queued.emit(data)
	return data


## Get the graveyard (deceased NPCs) as a raw Dictionary (use with [code]await[/code]).
func get_graveyard_async() -> Dictionary:
	var data := await _get_async("/story/graveyard")
	if data.is_empty():
		return {}

	graveyard_received.emit(data)
	return data


## Queue an NPC birth request and return the raw result (use with [code]await[/code]).
func queue_npc_birth_async(zone: String, role: String = "") -> Dictionary:
	var body: Dictionary = {"zone": zone}
	if role != "":
		body["role"] = role

	var data := await _post_async("/story/npc_birth_request", body)
	if data.is_empty():
		return {}

	npc_birth_queued.emit(data)
	return data


## Get the population stats as a raw Dictionary (use with [code]await[/code]).
func get_population_async() -> Dictionary:
	var data := await _get_async("/story/population")
	if data.is_empty():
		return {}

	population_received.emit(data)
	return data


## Refuse a quest and return the result (use with [code]await[/code]).
func refuse_quest_async(quest_id: String, npc_id: String, reason: String = "") -> NPCModels.QuestRefusalResult:
	var body: Dictionary = {"quest_id": quest_id, "npc_id": npc_id}
	if reason != "":
		body["reason"] = reason

	var data := await _post_async("/quests/refuse", body)
	if data.is_empty():
		return null

	var result := NPCModels.QuestRefusalResult.from_dict(data)
	quest_refused.emit(result)
	return result


## Set auto-refuse intents and return the result (use with [code]await[/code]).
func set_auto_refuse_async(intents: PackedStringArray) -> NPCModels.AutoRefuseResult:
	var intents_array: Array = []
	for intent in intents:
		intents_array.append(intent)

	var data := await _post_async("/player/auto_refuse", {"intents": intents_array})
	if data.is_empty():
		return null

	var result := NPCModels.AutoRefuseResult.from_dict(data)
	auto_refuse_received.emit(result)
	return result


## Get auto-refuse config (use with [code]await[/code]).
func get_auto_refuse_async() -> NPCModels.AutoRefuseResult:
	var data := await _get_async("/player/auto_refuse")
	if data.is_empty():
		return null

	var result := NPCModels.AutoRefuseResult.from_dict(data)
	auto_refuse_received.emit(result)
	return result


## Introduce the player to an NPC and return the result (use with [code]await[/code]).
func introduce_player_async(to_npc: String, player_name: String, titles: PackedStringArray = PackedStringArray()) -> NPCModels.IntroduceResult:
	var body: Dictionary = {"to_npc": to_npc, "name": player_name}
	if titles.size() > 0:
		var titles_array: Array = []
		for title in titles:
			titles_array.append(title)
		body["titles"] = titles_array

	var data := await _post_async("/player/introduce", body)
	if data.is_empty():
		return null

	var result := NPCModels.IntroduceResult.from_dict(data)
	player_introduced.emit(result)
	return result


## Set a visible feature and return the result (use with [code]await[/code]).
func set_visible_feature_async(feature: String) -> NPCModels.VisibleFeatureResult:
	var data := await _post_async("/player/visible_feature", {"feature": feature})
	if data.is_empty():
		return null

	var result := NPCModels.VisibleFeatureResult.from_dict(data)
	visible_feature_set.emit(result)
	return result


## Register a feature-identity mapping and return the result (use with [code]await[/code]).
func register_feature_async(feature: String, identity: String) -> NPCModels.RegisterFeatureResult:
	var data := await _post_async("/player/register_feature", {"feature": feature, "identity": identity})
	if data.is_empty():
		return null

	var result := NPCModels.RegisterFeatureResult.from_dict(data)
	feature_registered.emit(result)
	return result


## Vouch the player to an NPC and return the result (use with [code]await[/code]).
func vouch_player_async(voucher_npc: String, to_npc: String) -> NPCModels.VouchResult:
	var data := await _post_async("/player/vouched_by", {"voucher_npc": voucher_npc, "to_npc": to_npc})
	if data.is_empty():
		return null

	var result := NPCModels.VouchResult.from_dict(data)
	player_vouched.emit(result)
	return result


## Get the identity state as a raw Dictionary (use with [code]await[/code]).
func get_identity_state_async() -> Dictionary:
	var data := await _get_async("/player/identity_state")
	if data.is_empty():
		return {}

	identity_state_received.emit(data)
	return data


## Get the reputation as a raw Dictionary (use with [code]await[/code]).
func get_reputation_async() -> Dictionary:
	var data := await _get_async("/player/reputation")
	if data.is_empty():
		return {}

	reputation_received.emit(data)
	return data


# ---------------------------------------------------------------------------
# Signal-based response handlers
# ---------------------------------------------------------------------------

func _on_generate_response(data: Dictionary) -> void:
	var result := NPCModels.GenerateResult.from_dict(data)
	npc_response_received.emit(result)


func _on_list_npcs_response(data: Dictionary) -> void:
	var result := NPCModels.NPCListResult.from_dict(data)
	npc_list_received.emit(result)


func _on_switch_npc_response(data: Dictionary) -> void:
	var info := NPCModels.NPCInfo.from_dict(data)
	npc_switched.emit(info)


func _on_trust_response(data: Dictionary) -> void:
	var result := NPCModels.TrustResult.from_dict(data)
	trust_adjusted.emit(result)


func _on_mood_response(data: Dictionary) -> void:
	var result := NPCModels.MoodResult.from_dict(data)
	mood_set.emit(result)


func _on_inject_event_response(data: Dictionary) -> void:
	var result := NPCModels.EventResult.from_dict(data)
	event_injected.emit(result)


func _on_health_response(data: Dictionary) -> void:
	var result := NPCModels.HealthResult.from_dict(data)
	if result.status == "ok" or result.status == "healthy":
		if not _connected:
			_connected = true
			server_connected.emit()
		_consecutive_failures = 0


# ---------------------------------------------------------------------------
# Story Director — signal-based response handlers
# ---------------------------------------------------------------------------

func _on_story_reset_response(data: Dictionary) -> void:
	var result := NPCModels.StoryResetResult.from_dict(data)
	story_reset_completed.emit(result)


func _on_activity_response(data: Dictionary) -> void:
	var result := NPCModels.ActivityResult.from_dict(data)
	activity_received.emit(result)


func _on_pause_changed_response(data: Dictionary) -> void:
	var result := NPCModels.PauseResult.from_dict(data)
	pause_changed.emit(result)


func _on_tick_budget_response(data: Dictionary) -> void:
	var result := NPCModels.TickBudgetResult.from_dict(data)
	tick_budget_set.emit(result)


func _on_quest_pacing_response(data: Dictionary) -> void:
	var result := NPCModels.QuestPacingResult.from_dict(data)
	quest_pacing_received.emit(result)


func _on_npc_death_response(data: Dictionary) -> void:
	npc_death_queued.emit(data)


func _on_npc_birth_response(data: Dictionary) -> void:
	npc_birth_queued.emit(data)


func _on_quest_refused_response(data: Dictionary) -> void:
	var result := NPCModels.QuestRefusalResult.from_dict(data)
	quest_refused.emit(result)


func _on_auto_refuse_response(data: Dictionary) -> void:
	var result := NPCModels.AutoRefuseResult.from_dict(data)
	auto_refuse_received.emit(result)


func _on_player_introduced_response(data: Dictionary) -> void:
	var result := NPCModels.IntroduceResult.from_dict(data)
	player_introduced.emit(result)


func _on_visible_feature_response(data: Dictionary) -> void:
	var result := NPCModels.VisibleFeatureResult.from_dict(data)
	visible_feature_set.emit(result)


func _on_feature_registered_response(data: Dictionary) -> void:
	var result := NPCModels.RegisterFeatureResult.from_dict(data)
	feature_registered.emit(result)


func _on_player_vouched_response(data: Dictionary) -> void:
	var result := NPCModels.VouchResult.from_dict(data)
	player_vouched.emit(result)


# ---------------------------------------------------------------------------
# Internal HTTP helpers
# ---------------------------------------------------------------------------

## Perform an HTTP POST and invoke [param callback] with the parsed response Dictionary.
func _post(path: String, body: Dictionary, callback: Callable) -> void:
	var http := HTTPRequest.new()
	http.timeout = request_timeout
	add_child(http)

	var url := server_url + path
	var json_body := JSON.stringify(body)
	var headers := PackedStringArray(["Content-Type: application/json"])

	http.request_completed.connect(func(result: int, status_code: int, _headers: PackedStringArray, response_body: PackedByteArray) -> void:
		_handle_response(result, status_code, response_body, path, callback)
		http.queue_free()
	)

	var err := http.request(url, headers, HTTPClient.METHOD_POST, json_body)
	if err != OK:
		push_error("NPCEngineClient: Failed to send POST to %s (error %d)." % [url, err])
		request_failed.emit(path, "HTTPRequest.request() returned error %d" % err)
		http.queue_free()


## Perform an HTTP GET and invoke [param callback] with the parsed response Dictionary.
func _get(path: String, callback: Callable) -> void:
	var http := HTTPRequest.new()
	http.timeout = request_timeout
	add_child(http)

	var url := server_url + path

	http.request_completed.connect(func(result: int, status_code: int, _headers: PackedStringArray, response_body: PackedByteArray) -> void:
		_handle_response(result, status_code, response_body, path, callback)
		http.queue_free()
	)

	var err := http.request(url, [], HTTPClient.METHOD_GET)
	if err != OK:
		push_error("NPCEngineClient: Failed to send GET to %s (error %d)." % [url, err])
		request_failed.emit(path, "HTTPRequest.request() returned error %d" % err)
		http.queue_free()


## Parse the HTTP response and route to the callback or error signal.
func _handle_response(result: int, status_code: int, response_body: PackedByteArray, path: String, callback: Callable) -> void:
	## Connection-level failure.
	if result != HTTPRequest.RESULT_SUCCESS:
		_consecutive_failures += 1
		if _connected and _consecutive_failures >= 3:
			_connected = false
			server_disconnected.emit()
		var msg := "HTTP result code %d for %s" % [result, path]
		push_warning("NPCEngineClient: %s" % msg)
		request_failed.emit(path, msg)
		return

	## Reset failure counter on any successful connection.
	_consecutive_failures = 0

	## Non-2xx status code.
	if status_code < 200 or status_code >= 300:
		var body_text := response_body.get_string_from_utf8()
		var msg := "HTTP %d from %s: %s" % [status_code, path, body_text.left(256)]
		push_warning("NPCEngineClient: %s" % msg)
		request_failed.emit(path, msg)
		return

	## Parse JSON body.
	var body_text := response_body.get_string_from_utf8()
	var parsed = JSON.parse_string(body_text)
	if parsed == null or not (parsed is Dictionary):
		var msg := "Invalid JSON from %s: %s" % [path, body_text.left(256)]
		push_warning("NPCEngineClient: %s" % msg)
		request_failed.emit(path, msg)
		return

	callback.call(parsed as Dictionary)


# ---------------------------------------------------------------------------
# Await-style internal helpers
# ---------------------------------------------------------------------------

## POST that returns the parsed Dictionary (empty on failure).
func _post_async(path: String, body: Dictionary) -> Dictionary:
	var http := HTTPRequest.new()
	http.timeout = request_timeout
	add_child(http)

	var url := server_url + path
	var json_body := JSON.stringify(body)
	var headers := PackedStringArray(["Content-Type: application/json"])

	var err := http.request(url, headers, HTTPClient.METHOD_POST, json_body)
	if err != OK:
		push_error("NPCEngineClient: Failed to send POST to %s (error %d)." % [url, err])
		request_failed.emit(path, "HTTPRequest.request() returned error %d" % err)
		http.queue_free()
		return {}

	var response: Array = await http.request_completed
	http.queue_free()
	return _parse_async_response(response, path)


## GET that returns the parsed Dictionary (empty on failure).
func _get_async(path: String) -> Dictionary:
	var http := HTTPRequest.new()
	http.timeout = request_timeout
	add_child(http)

	var url := server_url + path

	var err := http.request(url, [], HTTPClient.METHOD_GET)
	if err != OK:
		push_error("NPCEngineClient: Failed to send GET to %s (error %d)." % [url, err])
		request_failed.emit(path, "HTTPRequest.request() returned error %d" % err)
		http.queue_free()
		return {}

	var response: Array = await http.request_completed
	http.queue_free()
	return _parse_async_response(response, path)


## Shared response parser for the async helpers.
func _parse_async_response(response: Array, path: String) -> Dictionary:
	var result: int = response[0]
	var status_code: int = response[1]
	var response_body: PackedByteArray = response[3]

	if result != HTTPRequest.RESULT_SUCCESS:
		_consecutive_failures += 1
		if _connected and _consecutive_failures >= 3:
			_connected = false
			server_disconnected.emit()
		var msg := "HTTP result code %d for %s" % [result, path]
		push_warning("NPCEngineClient: %s" % msg)
		request_failed.emit(path, msg)
		return {}

	_consecutive_failures = 0

	if status_code < 200 or status_code >= 300:
		var body_text := response_body.get_string_from_utf8()
		var msg := "HTTP %d from %s: %s" % [status_code, path, body_text.left(256)]
		push_warning("NPCEngineClient: %s" % msg)
		request_failed.emit(path, msg)
		return {}

	var body_text := response_body.get_string_from_utf8()
	var parsed = JSON.parse_string(body_text)
	if parsed == null or not (parsed is Dictionary):
		var msg := "Invalid JSON from %s: %s" % [path, body_text.left(256)]
		push_warning("NPCEngineClient: %s" % msg)
		request_failed.emit(path, msg)
		return {}

	return parsed as Dictionary
