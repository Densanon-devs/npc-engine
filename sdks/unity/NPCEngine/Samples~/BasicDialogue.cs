using UnityEngine;
using UnityEngine.UI;
using NPCEngine;

/// <summary>
/// Basic NPC dialogue example.
///
/// Setup:
///   1. Add NPCEngineServer and NPCEngineClient to this GameObject
///   2. Assign the UI fields in the Inspector
///   3. Press Play — the server starts automatically
///   4. Type in the input field and press Enter to talk to NPCs
///
/// The server auto-launches from StreamingAssets/NPCEngine/npc-engine.exe
/// (downloaded via Window → NPC Engine → Setup Wizard).
/// </summary>
public class BasicDialogue : MonoBehaviour
{
    [Header("UI")]
    public InputField playerInput;
    public Text dialogueText;
    public Text emotionText;
    public Text npcNameText;

    [Header("NPC")]
    public string activeNPC = "noah";

    private NPCEngineClient _client;

    void Start()
    {
        _client = GetComponent<NPCEngineClient>();

        // Wait for server to be ready before enabling input
        var server = GetComponent<NPCEngineServer>();
        if (server != null)
        {
            server.OnServerReady.AddListener(() =>
            {
                dialogueText.text = "NPC Engine ready! Type something to talk.";
                playerInput.interactable = true;
            });

            server.OnServerError.AddListener((error) =>
            {
                dialogueText.text = $"Server error: {error}";
            });
        }

        playerInput.onEndEdit.AddListener(OnPlayerSubmit);
        playerInput.interactable = false;
        dialogueText.text = "Starting NPC Engine...";
    }

    async void OnPlayerSubmit(string text)
    {
        if (string.IsNullOrEmpty(text)) return;
        if (!Input.GetKeyDown(KeyCode.Return)) return;

        playerInput.interactable = false;
        dialogueText.text = "...";

        var response = await _client.GenerateAsync(text, activeNPC);

        if (response != null && response.parsed != null)
        {
            // Display the NPC's dialogue
            dialogueText.text = response.parsed.dialogue;

            // Use the emotion for animation/UI
            if (emotionText != null)
                emotionText.text = $"[{response.parsed.emotion}]";

            // Show which NPC is talking
            if (npcNameText != null)
                npcNameText.text = response.npc_id;
        }
        else
        {
            dialogueText.text = "(No response from NPC Engine)";
        }

        playerInput.text = "";
        playerInput.interactable = true;
        playerInput.ActivateInputField();
    }

    /// <summary>
    /// Switch to a different NPC at runtime.
    /// Call this from a UI button or trigger.
    /// </summary>
    public async void SwitchNPC(string npcId)
    {
        activeNPC = npcId;
        var info = await _client.SwitchNPCAsync(npcId);
        if (info != null && npcNameText != null)
            npcNameText.text = info.name;
    }

    /// <summary>
    /// Inject a world event that all NPCs can perceive.
    /// Call this from gameplay triggers (e.g., dragon spawns, building destroyed).
    /// </summary>
    public async void InjectWorldEvent(string description)
    {
        await _client.InjectEventAsync(description);
    }
}
