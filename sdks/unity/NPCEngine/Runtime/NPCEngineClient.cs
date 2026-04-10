using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;

namespace NPCEngine
{
    /// <summary>
    /// HTTP client for the NPC Engine REST API.
    /// Attach to a GameObject and configure the server URL in the Inspector.
    /// All API methods are async and return typed response objects.
    /// </summary>
    public class NPCEngineClient : MonoBehaviour
    {
        [SerializeField]
        [Tooltip("Base URL of the NPC Engine server.")]
        private string serverUrl = "http://127.0.0.1:8000";

        /// <summary>
        /// Gets or sets the server URL at runtime.
        /// </summary>
        public string ServerUrl
        {
            get => serverUrl;
            set => serverUrl = value.TrimEnd('/');
        }

        // --------------------------------------------------------------------
        // Public API
        // --------------------------------------------------------------------

        /// <summary>
        /// Generates an NPC dialogue response for the given prompt.
        /// </summary>
        /// <param name="prompt">The player's input text.</param>
        /// <param name="npcId">Optional NPC ID to target. Null uses the active NPC.</param>
        /// <returns>A <see cref="GenerateResponse"/> with the raw and parsed dialogue.</returns>
        public async Task<GenerateResponse> GenerateAsync(string prompt, string npcId = null)
        {
            var body = npcId != null
                ? JsonUtility.ToJson(new GenerateRequest { prompt = prompt, npc_id = npcId })
                : JsonUtility.ToJson(new GenerateRequestSimple { prompt = prompt });

            string json = await PostAsync("/generate", body);
            if (json == null) return null;

            var result = JsonUtility.FromJson<GenerateResponse>(json);
            if (result != null && !string.IsNullOrEmpty(result.response))
            {
                NPCResponseParser.TryParse(result.response, out var parsed);
                result.parsed = parsed;
            }

            return result;
        }

        /// <summary>
        /// Lists all active NPCs in the current world.
        /// </summary>
        /// <returns>An <see cref="NPCListResponse"/> containing NPC info.</returns>
        public async Task<NPCListResponse> ListNPCsAsync()
        {
            string json = await GetAsync("/npcs");
            if (json == null) return null;
            return JsonUtility.FromJson<NPCListResponse>(json);
        }

        /// <summary>
        /// Switches the active NPC to the specified ID.
        /// </summary>
        /// <param name="npcId">The NPC ID to activate.</param>
        /// <returns>Info about the newly active NPC.</returns>
        public async Task<NPCInfo> SwitchNPCAsync(string npcId)
        {
            var body = JsonUtility.ToJson(new SwitchNPCRequest { npc_id = npcId });
            string json = await PostAsync("/npcs/switch", body);
            if (json == null) return null;
            return JsonUtility.FromJson<NPCInfo>(json);
        }

        /// <summary>
        /// Injects a world event that NPCs can perceive and react to.
        /// </summary>
        /// <param name="description">Description of the event.</param>
        /// <param name="npcId">Optional target NPC. Null broadcasts to all.</param>
        /// <returns>An <see cref="EventResponse"/> confirming injection.</returns>
        public async Task<EventResponse> InjectEventAsync(string description, string npcId = null)
        {
            var body = npcId != null
                ? JsonUtility.ToJson(new InjectEventRequest { event_description = description, npc_id = npcId })
                : JsonUtility.ToJson(new InjectEventRequestSimple { event_description = description });

            string json = await PostAsync("/event", body);
            if (json == null) return null;
            return JsonUtility.FromJson<EventResponse>(json);
        }

        /// <summary>
        /// Adjusts the trust level between the player and an NPC.
        /// </summary>
        /// <param name="npcId">The target NPC ID.</param>
        /// <param name="delta">Trust adjustment (positive or negative).</param>
        /// <param name="reason">Optional reason for the change.</param>
        /// <returns>A <see cref="TrustResponse"/> showing old and new levels.</returns>
        public async Task<TrustResponse> AdjustTrustAsync(string npcId, int delta, string reason = "")
        {
            var body = JsonUtility.ToJson(new AdjustTrustRequest
            {
                npc_id = npcId,
                delta = delta,
                reason = reason
            });

            string json = await PostAsync("/trust", body);
            if (json == null) return null;
            return JsonUtility.FromJson<TrustResponse>(json);
        }

        /// <summary>
        /// Sets the mood of an NPC.
        /// </summary>
        /// <param name="npcId">The target NPC ID.</param>
        /// <param name="mood">The mood to set (e.g., "happy", "angry", "fearful").</param>
        /// <param name="intensity">Mood intensity from 0.0 to 1.0.</param>
        /// <returns>A <see cref="MoodResponse"/> showing old and new mood.</returns>
        public async Task<MoodResponse> SetMoodAsync(string npcId, string mood, float intensity = 0.5f)
        {
            var body = JsonUtility.ToJson(new SetMoodRequest
            {
                npc_id = npcId,
                mood = mood,
                intensity = intensity
            });

            string json = await PostAsync("/mood", body);
            if (json == null) return null;
            return JsonUtility.FromJson<MoodResponse>(json);
        }

        /// <summary>
        /// Adds a note to an NPC's scratchpad memory.
        /// </summary>
        /// <param name="npcId">The target NPC ID.</param>
        /// <param name="text">The scratchpad text to add.</param>
        /// <param name="importance">Importance weight from 0.0 to 1.0.</param>
        public async Task AddScratchpadAsync(string npcId, string text, float importance = 0.7f)
        {
            var body = JsonUtility.ToJson(new AddScratchpadRequest
            {
                npc_id = npcId,
                text = text,
                importance = importance
            });

            await PostAsync("/scratchpad", body);
        }

        /// <summary>
        /// Marks a quest as accepted by the player.
        /// </summary>
        /// <param name="questId">Unique quest identifier.</param>
        /// <param name="questName">Display name of the quest.</param>
        /// <param name="givenBy">NPC ID who gave the quest.</param>
        public async Task AcceptQuestAsync(string questId, string questName, string givenBy)
        {
            var body = JsonUtility.ToJson(new AcceptQuestRequest
            {
                quest_id = questId,
                quest_name = questName,
                given_by = givenBy
            });

            await PostAsync("/quest/accept", body);
        }

        /// <summary>
        /// Marks a quest as completed.
        /// </summary>
        /// <param name="questId">The quest ID to complete.</param>
        public async Task CompleteQuestAsync(string questId)
        {
            var body = JsonUtility.ToJson(new CompleteQuestRequest { quest_id = questId });
            await PostAsync("/quest/complete", body);
        }

        /// <summary>
        /// Checks if the NPC Engine server is running and healthy.
        /// </summary>
        /// <returns>A <see cref="HealthResponse"/> with status and version info.</returns>
        public async Task<HealthResponse> HealthCheckAsync()
        {
            string json = await GetAsync("/health");
            if (json == null) return null;
            return JsonUtility.FromJson<HealthResponse>(json);
        }

        // --------------------------------------------------------------------
        // Internal HTTP helpers
        // --------------------------------------------------------------------

        /// <summary>
        /// Sends a POST request with a JSON body and returns the response text.
        /// </summary>
        internal async Task<string> PostAsync(string path, string json)
        {
            string url = serverUrl.TrimEnd('/') + path;
            byte[] bodyRaw = Encoding.UTF8.GetBytes(json);

            using (var request = new UnityWebRequest(url, UnityWebRequest.kHttpVerbPOST))
            {
                request.uploadHandler = new UploadHandlerRaw(bodyRaw);
                request.downloadHandler = new DownloadHandlerBuffer();
                request.SetRequestHeader("Content-Type", "application/json");

                var operation = request.SendWebRequest();
                while (!operation.isDone)
                    await Task.Yield();

                if (request.result != UnityWebRequest.Result.Success)
                {
                    Debug.LogError($"[NPCEngine] POST {path} failed: {request.error}");
                    return null;
                }

                return request.downloadHandler.text;
            }
        }

        /// <summary>
        /// Sends a GET request and returns the response text.
        /// </summary>
        internal async Task<string> GetAsync(string path)
        {
            string url = serverUrl.TrimEnd('/') + path;

            using (var request = UnityWebRequest.Get(url))
            {
                var operation = request.SendWebRequest();
                while (!operation.isDone)
                    await Task.Yield();

                if (request.result != UnityWebRequest.Result.Success)
                {
                    Debug.LogError($"[NPCEngine] GET {path} failed: {request.error}");
                    return null;
                }

                return request.downloadHandler.text;
            }
        }

        // --------------------------------------------------------------------
        // Internal request DTOs (not exposed publicly)
        // --------------------------------------------------------------------

        [System.Serializable]
        private class GenerateRequest
        {
            public string prompt;
            public string npc_id;
        }

        [System.Serializable]
        private class GenerateRequestSimple
        {
            public string prompt;
        }

        [System.Serializable]
        private class SwitchNPCRequest
        {
            public string npc_id;
        }

        [System.Serializable]
        private class InjectEventRequest
        {
            public string event_description;
            public string npc_id;
        }

        [System.Serializable]
        private class InjectEventRequestSimple
        {
            public string event_description;
        }

        [System.Serializable]
        private class AdjustTrustRequest
        {
            public string npc_id;
            public int delta;
            public string reason;
        }

        [System.Serializable]
        private class SetMoodRequest
        {
            public string npc_id;
            public string mood;
            public float intensity;
        }

        [System.Serializable]
        private class AddScratchpadRequest
        {
            public string npc_id;
            public string text;
            public float importance;
        }

        [System.Serializable]
        private class AcceptQuestRequest
        {
            public string quest_id;
            public string quest_name;
            public string given_by;
        }

        [System.Serializable]
        private class CompleteQuestRequest
        {
            public string quest_id;
        }
    }
}
