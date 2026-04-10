@tool
extends EditorPlugin
## EditorPlugin that registers the NPCEngine autoload singleton.
##
## When the plugin is enabled in Project Settings > Plugins, it automatically
## adds the NPCEngineClient as a global autoload named "NPCEngine".


func _enter_tree() -> void:
	add_autoload_singleton("NPCEngine", "res://addons/npc_engine/autoload/npc_engine_client.gd")


func _exit_tree() -> void:
	remove_autoload_singleton("NPCEngine")
