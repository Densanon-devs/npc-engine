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
            string json = await GetAsync("/npc/list");
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
            string json = await PostAsync("/npc/switch", body);
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
                ? JsonUtility.ToJson(new InjectEventRequest { description = description, npc_id = npcId })
                : JsonUtility.ToJson(new InjectEventRequestSimple { description = description });

            string json = await PostAsync("/events/inject", body);
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

            string json = await PostAsync("/npc/trust", body);
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
        public async Task<MoodResponse> SetMoodAsync(string npcId, string mood, float intensity = 0.5f, int pinTurns = 3)
        {
            var body = JsonUtility.ToJson(new SetMoodRequest
            {
                npc_id = npcId,
                mood = mood,
                intensity = intensity,
                pin_turns = pinTurns
            });

            string json = await PostAsync("/npc/mood", body);
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

            await PostAsync("/npc/scratchpad", body);
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

            await PostAsync("/quests/accept", body);
        }

        /// <summary>
        /// Marks a quest as completed.
        /// </summary>
        /// <param name="questId">The quest ID to complete.</param>
        public async Task CompleteQuestAsync(string questId)
        {
            var body = JsonUtility.ToJson(new CompleteQuestRequest { quest_id = questId });
            await PostAsync("/quests/complete", body);
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
        // Story Director API
        // --------------------------------------------------------------------

        /// <summary>
        /// Advances the story world by one overseer tick.
        /// </summary>
        /// <returns>A <see cref="StoryTickResponse"/> with the tick result and timing hint.</returns>
        public async Task<StoryTickResponse> StoryTickAsync()
        {
            string json = await PostAsync("/story/tick", "{}");
            if (json == null) return null;
            return JsonUtility.FromJson<StoryTickResponse>(json);
        }

        /// <summary>
        /// Resets the story to the YAML baseline (born NPCs removed, deceased revived, state cleared).
        /// </summary>
        /// <returns>A <see cref="StoryResetResponse"/> with reset details.</returns>
        public async Task<StoryResetResponse> ResetStoryAsync()
        {
            string json = await PostAsync("/story/reset", "{}");
            if (json == null) return null;
            return JsonUtility.FromJson<StoryResetResponse>(json);
        }

        /// <summary>
        /// Sets the player's current activity context for the Story Director.
        /// </summary>
        /// <param name="activity">Activity string (e.g., "in_town", "in_combat", "idle").</param>
        /// <returns>An <see cref="ActivityResponse"/> confirming the activity.</returns>
        public async Task<ActivityResponse> SetActivityAsync(string activity)
        {
            var body = JsonUtility.ToJson(new SetActivityRequest { activity = activity });
            string json = await PostAsync("/story/activity", body);
            if (json == null) return null;
            return JsonUtility.FromJson<ActivityResponse>(json);
        }

        /// <summary>
        /// Gets the player's current activity context.
        /// </summary>
        /// <returns>An <see cref="ActivityResponse"/> with the current activity.</returns>
        public async Task<ActivityResponse> GetActivityAsync()
        {
            string json = await GetAsync("/story/activity");
            if (json == null) return null;
            return JsonUtility.FromJson<ActivityResponse>(json);
        }

        /// <summary>
        /// Pauses all future story ticks.
        /// </summary>
        /// <returns>A <see cref="PauseResponse"/> confirming the pause.</returns>
        public async Task<PauseResponse> PauseStoryAsync()
        {
            string json = await PostAsync("/story/pause", "{}");
            if (json == null) return null;
            return JsonUtility.FromJson<PauseResponse>(json);
        }

        /// <summary>
        /// Resumes story ticks after a pause.
        /// </summary>
        /// <returns>A <see cref="PauseResponse"/> confirming the resume.</returns>
        public async Task<PauseResponse> ResumeStoryAsync()
        {
            string json = await PostAsync("/story/resume", "{}");
            if (json == null) return null;
            return JsonUtility.FromJson<PauseResponse>(json);
        }

        /// <summary>
        /// Gets the current pause state including budget info and next-tick hint.
        /// </summary>
        /// <returns>A <see cref="PauseStateResponse"/> with full pause/budget details.</returns>
        public async Task<PauseStateResponse> GetPauseStateAsync()
        {
            string json = await GetAsync("/story/pause_state");
            if (json == null) return null;
            return JsonUtility.FromJson<PauseStateResponse>(json);
        }

        /// <summary>
        /// Sets the rolling-window LLM time budget for story ticks.
        /// </summary>
        /// <param name="maxSecondsPerMinute">Max LLM seconds per minute. Negative clears the budget.</param>
        /// <returns>A <see cref="TickBudgetResponse"/> confirming the budget.</returns>
        public async Task<TickBudgetResponse> SetTickBudgetAsync(float maxSecondsPerMinute)
        {
            var body = JsonUtility.ToJson(new SetTickBudgetRequest { max_seconds_per_minute = maxSecondsPerMinute });
            string json = await PostAsync("/story/tick_budget", body);
            if (json == null) return null;
            return JsonUtility.FromJson<TickBudgetResponse>(json);
        }

        // --------------------------------------------------------------------
        // Quest Pacing API
        // --------------------------------------------------------------------

        /// <summary>
        /// Overrides per-NPC quest pacing caps.
        /// </summary>
        /// <param name="maxUnoffered">Max unoffered quests per NPC. Null leaves unchanged.</param>
        /// <param name="cooldownTicks">Cooldown ticks between quest dispatches. Null leaves unchanged.</param>
        /// <returns>A <see cref="QuestPacingResponse"/> with effective pacing values.</returns>
        public async Task<QuestPacingResponse> SetQuestPacingAsync(int? maxUnoffered = null, int? cooldownTicks = null)
        {
            // Build JSON manually to omit null fields
            var parts = new System.Collections.Generic.List<string>();
            if (maxUnoffered.HasValue)
                parts.Add($"\"max_unoffered\":{maxUnoffered.Value}");
            if (cooldownTicks.HasValue)
                parts.Add($"\"cooldown_ticks\":{cooldownTicks.Value}");
            string body = "{" + string.Join(",", parts) + "}";

            string json = await PostAsync("/story/quest_pacing", body);
            if (json == null) return null;
            return JsonUtility.FromJson<QuestPacingResponse>(json);
        }

        /// <summary>
        /// Gets the effective quest pacing configuration.
        /// </summary>
        /// <returns>A <see cref="QuestPacingResponse"/> with current pacing values.</returns>
        public async Task<QuestPacingResponse> GetQuestPacingAsync()
        {
            string json = await GetAsync("/story/quest_pacing");
            if (json == null) return null;
            return JsonUtility.FromJson<QuestPacingResponse>(json);
        }

        // --------------------------------------------------------------------
        // NPC Lifecycle API
        // --------------------------------------------------------------------

        /// <summary>
        /// Queues a death dispatch for an NPC (game-authoritative).
        /// </summary>
        /// <param name="npcId">The NPC ID to kill.</param>
        /// <param name="cause">Optional cause of death.</param>
        /// <param name="transfersQuestsTo">Optional NPC ID to inherit open quests.</param>
        /// <returns>A <see cref="GenericOkResponse"/> confirming the queue.</returns>
        public async Task<GenericOkResponse> QueueNPCDeathAsync(string npcId, string cause = "", string transfersQuestsTo = null)
        {
            // Build JSON manually to omit null transfers_quests_to
            string body;
            if (transfersQuestsTo != null)
            {
                body = JsonUtility.ToJson(new NPCDeathRequestFull
                {
                    npc_id = npcId,
                    cause = cause,
                    transfers_quests_to = transfersQuestsTo
                });
            }
            else
            {
                body = JsonUtility.ToJson(new NPCDeathRequest
                {
                    npc_id = npcId,
                    cause = cause
                });
            }

            string json = await PostAsync("/story/npc_death", body);
            if (json == null) return null;
            return JsonUtility.FromJson<GenericOkResponse>(json);
        }

        /// <summary>
        /// Gets the graveyard (deceased NPCs and their death records).
        /// </summary>
        /// <returns>Raw JSON string (complex nested structure).</returns>
        public async Task<string> GetGraveyardAsync()
        {
            return await GetAsync("/story/graveyard");
        }

        /// <summary>
        /// Queues a birth request for a new NPC in a zone.
        /// </summary>
        /// <param name="zone">The zone to spawn the NPC in.</param>
        /// <param name="role">Optional role for the new NPC.</param>
        /// <returns>A <see cref="GenericOkResponse"/> confirming the queue.</returns>
        public async Task<GenericOkResponse> QueueNPCBirthAsync(string zone, string role = null)
        {
            // Build JSON manually to omit null role
            string body;
            if (role != null)
            {
                body = JsonUtility.ToJson(new NPCBirthRequestFull
                {
                    zone = zone,
                    role = role
                });
            }
            else
            {
                body = JsonUtility.ToJson(new NPCBirthRequestSimple { zone = zone });
            }

            string json = await PostAsync("/story/npc_birth_request", body);
            if (json == null) return null;
            return JsonUtility.FromJson<GenericOkResponse>(json);
        }

        /// <summary>
        /// Gets per-zone population counts.
        /// </summary>
        /// <returns>Raw JSON string (complex nested structure).</returns>
        public async Task<string> GetPopulationAsync()
        {
            return await GetAsync("/story/population");
        }

        // --------------------------------------------------------------------
        // Quest Refusal API
        // --------------------------------------------------------------------

        /// <summary>
        /// Player refuses a quest (trust hit, ledger entry, decay timer if applicable).
        /// </summary>
        /// <param name="questId">The quest ID to refuse.</param>
        /// <param name="npcId">The NPC who offered the quest.</param>
        /// <param name="reason">Optional refusal reason.</param>
        /// <returns>A <see cref="QuestRefusalResponse"/> with refusal details.</returns>
        public async Task<QuestRefusalResponse> RefuseQuestAsync(string questId, string npcId, string reason = null)
        {
            // Build JSON manually to omit null reason
            string body;
            if (reason != null)
            {
                body = JsonUtility.ToJson(new RefuseQuestRequestFull
                {
                    quest_id = questId,
                    npc_id = npcId,
                    reason = reason
                });
            }
            else
            {
                body = JsonUtility.ToJson(new RefuseQuestRequest
                {
                    quest_id = questId,
                    npc_id = npcId
                });
            }

            string json = await PostAsync("/quests/refuse", body);
            if (json == null) return null;
            return JsonUtility.FromJson<QuestRefusalResponse>(json);
        }

        /// <summary>
        /// Sets the player's auto-refuse intent filter.
        /// </summary>
        /// <param name="intents">Array of intent tags to auto-refuse.</param>
        /// <returns>An <see cref="AutoRefuseResponse"/> confirming the config.</returns>
        public async Task<AutoRefuseResponse> SetAutoRefuseAsync(string[] intents)
        {
            var body = JsonUtility.ToJson(new SetAutoRefuseRequest { intents = intents });
            string json = await PostAsync("/player/auto_refuse", body);
            if (json == null) return null;
            return JsonUtility.FromJson<AutoRefuseResponse>(json);
        }

        /// <summary>
        /// Gets the current auto-refuse configuration.
        /// </summary>
        /// <returns>An <see cref="AutoRefuseResponse"/> with current config.</returns>
        public async Task<AutoRefuseResponse> GetAutoRefuseAsync()
        {
            string json = await GetAsync("/player/auto_refuse");
            if (json == null) return null;
            return JsonUtility.FromJson<AutoRefuseResponse>(json);
        }

        // --------------------------------------------------------------------
        // Player Identity API
        // --------------------------------------------------------------------

        /// <summary>
        /// Player introduces themselves to an NPC.
        /// </summary>
        /// <param name="toNpc">The NPC ID to introduce to.</param>
        /// <param name="name">The player's name.</param>
        /// <param name="titles">Optional titles to share.</param>
        /// <returns>An <see cref="IntroduceResponse"/> with recognition details.</returns>
        public async Task<IntroduceResponse> IntroducePlayerAsync(string toNpc, string name, string[] titles = null)
        {
            // Build JSON manually to omit null titles
            string body;
            if (titles != null)
            {
                body = JsonUtility.ToJson(new IntroduceRequestFull
                {
                    to_npc = toNpc,
                    name = name,
                    titles = titles
                });
            }
            else
            {
                body = JsonUtility.ToJson(new IntroduceRequest
                {
                    to_npc = toNpc,
                    name = name
                });
            }

            string json = await PostAsync("/player/introduce", body);
            if (json == null) return null;
            return JsonUtility.FromJson<IntroduceResponse>(json);
        }

        /// <summary>
        /// Sets a player-visible feature (e.g., cloak, weapon).
        /// </summary>
        /// <param name="feature">The visible feature string.</param>
        /// <returns>A <see cref="VisibleFeatureResponse"/> confirming the feature.</returns>
        public async Task<VisibleFeatureResponse> SetVisibleFeatureAsync(string feature)
        {
            var body = JsonUtility.ToJson(new SetVisibleFeatureRequest { feature = feature });
            string json = await PostAsync("/player/visible_feature", body);
            if (json == null) return null;
            return JsonUtility.FromJson<VisibleFeatureResponse>(json);
        }

        /// <summary>
        /// Maps a visible feature to an identity for auto-recognition.
        /// </summary>
        /// <param name="feature">The visible feature to register.</param>
        /// <param name="identity">The identity this feature maps to.</param>
        /// <returns>A <see cref="RegisterFeatureResponse"/> confirming the mapping.</returns>
        public async Task<RegisterFeatureResponse> RegisterFeatureAsync(string feature, string identity)
        {
            var body = JsonUtility.ToJson(new RegisterFeatureRequest
            {
                feature = feature,
                identity = identity
            });
            string json = await PostAsync("/player/register_feature", body);
            if (json == null) return null;
            return JsonUtility.FromJson<RegisterFeatureResponse>(json);
        }

        /// <summary>
        /// One NPC vouches the player to another NPC (identity inheritance).
        /// </summary>
        /// <param name="voucherNpc">The NPC doing the vouching.</param>
        /// <param name="toNpc">The NPC being vouched to.</param>
        /// <returns>A <see cref="VouchResponse"/> with vouch details.</returns>
        public async Task<VouchResponse> VouchPlayerAsync(string voucherNpc, string toNpc)
        {
            var body = JsonUtility.ToJson(new VouchRequest
            {
                voucher_npc = voucherNpc,
                to_npc = toNpc
            });
            string json = await PostAsync("/player/vouched_by", body);
            if (json == null) return null;
            return JsonUtility.FromJson<VouchResponse>(json);
        }

        /// <summary>
        /// Gets per-NPC player knowledge, identity trust, and feature registry.
        /// </summary>
        /// <returns>Raw JSON string (complex nested structure).</returns>
        public async Task<string> GetIdentityStateAsync()
        {
            return await GetAsync("/player/identity_state");
        }

        /// <summary>
        /// Gets aggregated per-identity reputation data.
        /// </summary>
        /// <returns>Raw JSON string (complex nested structure).</returns>
        public async Task<string> GetReputationAsync()
        {
            return await GetAsync("/player/reputation");
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
            public string description;
            public string npc_id;
        }

        [System.Serializable]
        private class InjectEventRequestSimple
        {
            public string description;
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
            public int pin_turns;
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

        // Story Director request DTOs

        [System.Serializable]
        private class SetActivityRequest
        {
            public string activity;
        }

        [System.Serializable]
        private class SetTickBudgetRequest
        {
            public float max_seconds_per_minute;
        }

        // NPC Lifecycle request DTOs

        [System.Serializable]
        private class NPCDeathRequest
        {
            public string npc_id;
            public string cause;
        }

        [System.Serializable]
        private class NPCDeathRequestFull
        {
            public string npc_id;
            public string cause;
            public string transfers_quests_to;
        }

        [System.Serializable]
        private class NPCBirthRequestSimple
        {
            public string zone;
        }

        [System.Serializable]
        private class NPCBirthRequestFull
        {
            public string zone;
            public string role;
        }

        // Quest Refusal request DTOs

        [System.Serializable]
        private class RefuseQuestRequest
        {
            public string quest_id;
            public string npc_id;
        }

        [System.Serializable]
        private class RefuseQuestRequestFull
        {
            public string quest_id;
            public string npc_id;
            public string reason;
        }

        [System.Serializable]
        private class SetAutoRefuseRequest
        {
            public string[] intents;
        }

        // Player Identity request DTOs

        [System.Serializable]
        private class IntroduceRequest
        {
            public string to_npc;
            public string name;
        }

        [System.Serializable]
        private class IntroduceRequestFull
        {
            public string to_npc;
            public string name;
            public string[] titles;
        }

        [System.Serializable]
        private class SetVisibleFeatureRequest
        {
            public string feature;
        }

        [System.Serializable]
        private class RegisterFeatureRequest
        {
            public string feature;
            public string identity;
        }

        [System.Serializable]
        private class VouchRequest
        {
            public string voucher_npc;
            public string to_npc;
        }
    }
}
