using UnityEngine;

namespace NPCEngine
{
    /// <summary>
    /// Project-wide settings for the NPC Engine SDK.
    /// Create via Assets > Create > NPC Engine > Settings.
    /// </summary>
    [CreateAssetMenu(fileName = "NPCEngineSettings", menuName = "NPC Engine/Settings", order = 1)]
    public class NPCEngineSettings : ScriptableObject
    {
        [Header("Server")]
        [Tooltip("Base URL of the NPC Engine server.")]
        public string serverUrl = "http://127.0.0.1:8000";

        [Tooltip("Path to the NPC Engine server binary. Leave empty for auto-detection.")]
        public string serverBinaryPath = "";

        [Tooltip("Automatically start the server when entering Play mode.")]
        public bool autoStartServer = true;

        [Header("Authentication")]
        [Tooltip("License key for NPC Engine (if required).")]
        public string licenseKey = "";
    }
}
