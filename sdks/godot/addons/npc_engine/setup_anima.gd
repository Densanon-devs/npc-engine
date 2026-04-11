@tool
extends EditorScript
## Anima Setup — Downloads the server binary and AI model.
##
## Run from the Godot editor: Script Editor > File > Run (or Ctrl+Shift+X)
##
## Downloads to res://addons/npc_engine/bin/ so the server manager can find it.
## After download, set NPCEngineServerManager.server_binary_path to:
##   "res://addons/npc_engine/bin/npc-engine" (Linux/macOS)
##   "res://addons/npc_engine/bin/npc-engine.exe" (Windows)

const BIN_DIR = "res://addons/npc_engine/bin/"
const MODEL_DIR = "res://addons/npc_engine/bin/models/"

# Update these URLs when new releases are published
const SERVER_URLS = {
	"windows": "https://github.com/Densanon-devs/npc-engine/releases/latest/download/NPCEngine-server-windows-latest.zip",
	"linux": "https://github.com/Densanon-devs/npc-engine/releases/latest/download/NPCEngine-server-linux-latest.zip",
	"macos": "https://github.com/Densanon-devs/npc-engine/releases/latest/download/NPCEngine-server-macos-latest.zip",
}

const MODEL_URL = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
const MODEL_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"


func _run() -> void:
	print("")
	print("  ╔══════════════════════════════════════════╗")
	print("  ║  Anima Setup — Every NPC has a soul      ║")
	print("  ╚══════════════════════════════════════════╝")
	print("")

	# Detect platform
	var platform := _get_platform()
	print("  Platform: %s" % platform)

	# Check what's already installed
	var bin_path := BIN_DIR + _get_binary_name()
	var model_path := MODEL_DIR + MODEL_FILE
	var has_server := FileAccess.file_exists(bin_path)
	var has_model := FileAccess.file_exists(model_path)

	print("  Server binary: %s" % ("INSTALLED" if has_server else "NOT FOUND"))
	print("  AI model:      %s" % ("INSTALLED" if has_model else "NOT FOUND"))
	print("")

	if has_server and has_model:
		print("  Anima is ready!")
		print("  Set NPCEngineServerManager.server_binary_path to:")
		print("    %s" % bin_path)
		print("")
		return

	# Create directories
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(BIN_DIR))
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(MODEL_DIR))

	if not has_server:
		print("  To download the Anima server binary:")
		print("    1. Go to: https://github.com/Densanon-devs/npc-engine/releases")
		print("    2. Download NPCEngine-server-%s-*.zip" % platform)
		print("    3. Extract into: %s" % ProjectSettings.globalize_path(BIN_DIR))
		print("")

	if not has_model:
		print("  To download the AI model (469MB):")
		print("    1. Go to: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF")
		print("    2. Download: %s" % MODEL_FILE)
		print("    3. Place in: %s" % ProjectSettings.globalize_path(MODEL_DIR))
		print("")

	print("  After downloading, set NPCEngineServerManager.server_binary_path to:")
	print("    %s" % bin_path)
	print("")
	print("  Then add NPCEngineServerManager + NPCEngineClient nodes to your scene.")
	print("  Anima will launch automatically when you run your game.")


func _get_platform() -> String:
	var os_name := OS.get_name().to_lower()
	if "windows" in os_name:
		return "windows"
	elif "mac" in os_name or "osx" in os_name:
		return "macos"
	else:
		return "linux"


func _get_binary_name() -> String:
	if OS.get_name().to_lower().contains("windows"):
		return "npc-engine.exe"
	return "npc-engine"
