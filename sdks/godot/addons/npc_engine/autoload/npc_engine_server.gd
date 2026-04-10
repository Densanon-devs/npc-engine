class_name NPCEngineServerManager
extends Node
## Manages the NPC Engine server binary as a subprocess.
##
## Add this node to your scene tree (or use it alongside the NPCEngineClient autoload)
## to automatically start and stop the NPC Engine server process. The server is
## started on [code]_ready()[/code] if [member auto_start] is enabled, and is
## killed when the game window closes.
##
## [codeblock]
## var server_manager := NPCEngineServerManager.new()
## server_manager.server_binary_path = "res://bin/npc_engine"
## server_manager.port = 8000
## add_child(server_manager)
## await server_manager.server_ready
## [/codeblock]

## Emitted when the server process is running and responds to health checks.
signal server_ready

## Emitted when the server fails to start or crashes.
signal server_error(message: String)

## Path to the NPC Engine server executable.
@export var server_binary_path: String = ""

## Port number the server listens on.
@export var port: int = 8000

## Whether to automatically start the server in [method _ready].
@export var auto_start: bool = true

## Maximum number of health-check attempts before giving up.
@export var health_check_retries: int = 20

## Delay in seconds between health-check polls.
@export var health_check_interval: float = 0.5

## Whether the server process is currently running.
var is_running: bool = false

## PID of the server process, or -1 if not running.
var _pid: int = -1

## HTTPRequest node used to poll the health endpoint.
var _health_request: HTTPRequest = null


func _ready() -> void:
	if auto_start and server_binary_path != "":
		start_server()


func _notification(what: int) -> void:
	if what == NOTIFICATION_WM_CLOSE_REQUEST:
		stop_server()


## Start the server binary as a detached process.
func start_server() -> void:
	if is_running:
		push_warning("NPCEngineServerManager: Server is already running (PID %d)." % _pid)
		return

	var resolved_path: String = ProjectSettings.globalize_path(server_binary_path)
	if not FileAccess.file_exists(resolved_path) and not FileAccess.file_exists(server_binary_path):
		var msg := "NPCEngineServerManager: Server binary not found at '%s'." % server_binary_path
		push_error(msg)
		server_error.emit(msg)
		return

	var args: PackedStringArray = PackedStringArray([
		"--port", str(port),
	])

	_pid = OS.create_process(resolved_path, args)
	if _pid <= 0:
		var msg := "NPCEngineServerManager: Failed to create process for '%s'." % resolved_path
		push_error(msg)
		server_error.emit(msg)
		return

	is_running = true
	_poll_health()


## Stop the server process if it is running.
func stop_server() -> void:
	if not is_running or _pid <= 0:
		return

	var err := OS.kill(_pid)
	if err != OK:
		push_warning("NPCEngineServerManager: Failed to kill PID %d (error %d)." % [_pid, err])

	_pid = -1
	is_running = false


## Poll the server health endpoint until it responds or retries are exhausted.
func _poll_health() -> void:
	if _health_request == null:
		_health_request = HTTPRequest.new()
		add_child(_health_request)

	var url := "http://127.0.0.1:%d/health" % port

	for i in health_check_retries:
		## Wait between polls.
		await get_tree().create_timer(health_check_interval).timeout

		if not is_running:
			var msg := "NPCEngineServerManager: Server process exited before becoming ready."
			push_error(msg)
			server_error.emit(msg)
			return

		_health_request.cancel_request()
		var err := _health_request.request(url, [], HTTPClient.METHOD_GET)
		if err != OK:
			continue

		var response: Array = await _health_request.request_completed
		var result_code: int = response[0]
		var status_code: int = response[1]

		if result_code == HTTPRequest.RESULT_SUCCESS and status_code == 200:
			server_ready.emit()
			return

	## Exhausted retries.
	var msg := "NPCEngineServerManager: Server did not respond to health checks after %d attempts." % health_check_retries
	push_error(msg)
	server_error.emit(msg)
