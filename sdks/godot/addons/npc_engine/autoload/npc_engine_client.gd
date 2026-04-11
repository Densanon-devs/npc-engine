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
func set_mood(npc_id: String, mood: String, intensity: float = 0.5) -> void:
	_post("/npc/mood", {"npc_id": npc_id, "mood": mood, "intensity": intensity}, _on_mood_response)


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
