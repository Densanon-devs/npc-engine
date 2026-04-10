using System.Threading.Tasks;
using UnityEngine;

namespace NPCEngine.Examples
{
    /// <summary>
    /// Minimal example showing how to use the NPC Engine client.
    /// Attach to a GameObject alongside <see cref="NPCEngineClient"/>.
    /// </summary>
    public class NPCEngineExample : MonoBehaviour
    {
        [SerializeField]
        [Tooltip("Reference to the NPCEngineClient component.")]
        private NPCEngineClient client;

        [SerializeField]
        [Tooltip("Key to press to send a test prompt.")]
        private KeyCode sendKey = KeyCode.Return;

        [SerializeField]
        [Tooltip("Test prompt to send when the key is pressed.")]
        private string testPrompt = "Hello, what can you tell me about this village?";

        private bool _isProcessing;

        private async void Start()
        {
            if (client == null)
            {
                client = GetComponent<NPCEngineClient>();
                if (client == null)
                {
                    Debug.LogError("[NPCEngineExample] No NPCEngineClient found. Assign one in the Inspector.");
                    return;
                }
            }

            // List all NPCs and switch to the first one
            await InitializeNPCs();
        }

        private void Update()
        {
            if (Input.GetKeyDown(sendKey) && !_isProcessing)
            {
                _ = SendPrompt(testPrompt);
            }
        }

        /// <summary>
        /// Lists active NPCs and switches to the first available one.
        /// </summary>
        private async Task InitializeNPCs()
        {
            Debug.Log("[NPCEngineExample] Listing NPCs...");

            var list = await client.ListNPCsAsync();
            if (list == null || list.npcs == null || list.npcs.Length == 0)
            {
                Debug.LogWarning("[NPCEngineExample] No NPCs available.");
                return;
            }

            Debug.Log($"[NPCEngineExample] Found {list.active} active NPC(s) in world '{list.world_name}':");
            foreach (var npc in list.npcs)
            {
                Debug.Log($"  - {npc.name} ({npc.id}): {npc.role}");
            }

            // Switch to the first NPC
            var first = list.npcs[0];
            Debug.Log($"[NPCEngineExample] Switching to {first.name}...");

            var info = await client.SwitchNPCAsync(first.id);
            if (info != null)
            {
                Debug.Log($"[NPCEngineExample] Active NPC: {info.name} ({info.role})");
            }
        }

        /// <summary>
        /// Sends a prompt to the active NPC and logs the dialogue response.
        /// </summary>
        private async Task SendPrompt(string prompt)
        {
            _isProcessing = true;

            Debug.Log($"[NPCEngineExample] Sending: \"{prompt}\"");

            var response = await client.GenerateAsync(prompt);
            if (response == null)
            {
                Debug.LogWarning("[NPCEngineExample] No response received.");
                _isProcessing = false;
                return;
            }

            Debug.Log($"[NPCEngineExample] NPC '{response.npc_id}' responded in {response.generation_time:F2}s");

            if (response.parsed != null)
            {
                Debug.Log($"[NPCEngineExample] Dialogue: {response.parsed.dialogue}");
                Debug.Log($"[NPCEngineExample] Emotion: {response.parsed.emotion}");

                if (!string.IsNullOrEmpty(response.parsed.action))
                    Debug.Log($"[NPCEngineExample] Action: {response.parsed.action}");

                if (response.parsed.quest != null)
                    Debug.Log($"[NPCEngineExample] Quest offered: {response.parsed.quest.objective}");
            }
            else
            {
                Debug.Log($"[NPCEngineExample] Raw response: {response.response}");
            }

            _isProcessing = false;
        }
    }
}
