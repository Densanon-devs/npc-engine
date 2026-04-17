using System;

namespace NPCEngine
{
    /// <summary>
    /// Quest data attached to an NPC dialogue response.
    /// </summary>
    [Serializable]
    public class QuestData
    {
        public string type;
        public string objective;
        public string reward;
    }

    /// <summary>
    /// Parsed dialogue content from an NPC response.
    /// Contains the dialogue text, emotional state, action, and optional quest.
    /// </summary>
    [Serializable]
    public class NPCDialogueContent
    {
        public string dialogue;
        public string emotion;
        public string action;
        public QuestData quest;
    }

    /// <summary>
    /// Raw response from the /generate endpoint.
    /// The <see cref="response"/> field contains a JSON string that must be parsed
    /// separately into <see cref="NPCDialogueContent"/>.
    /// </summary>
    [Serializable]
    public class GenerateResponse
    {
        public string npc_id;
        public string response;
        public float generation_time;

        /// <summary>
        /// Populated client-side after parsing the double-encoded JSON in <see cref="response"/>.
        /// </summary>
        [NonSerialized]
        public NPCDialogueContent parsed;
    }

    /// <summary>
    /// Information about a single NPC registered in the engine.
    /// </summary>
    [Serializable]
    public class NPCInfo
    {
        public string id;
        public string name;
        public string role;
        public string[] capabilities;
    }

    /// <summary>
    /// Response from the /npcs endpoint listing all active NPCs.
    /// </summary>
    [Serializable]
    public class NPCListResponse
    {
        public int active;
        public string world_name;
        public NPCInfo[] npcs;
    }

    /// <summary>
    /// Response from the /trust endpoint after adjusting an NPC's trust level.
    /// </summary>
    [Serializable]
    public class TrustResponse
    {
        public string npc_id;
        public string old_level;
        public string new_level;
        public int delta;
        public string reason;
    }

    /// <summary>
    /// Response from the /mood endpoint after setting an NPC's mood.
    /// </summary>
    [Serializable]
    public class MoodResponse
    {
        public string npc_id;
        public string old_mood;
        public string new_mood;
        public float intensity;
    }

    /// <summary>
    /// Response from the /event endpoint after injecting a world event.
    /// </summary>
    [Serializable]
    public class EventResponse
    {
        public bool injected;
        public string target;
        public string event_description;
    }

    /// <summary>
    /// Response from the /health endpoint.
    /// </summary>
    [Serializable]
    public class HealthResponse
    {
        public string status;
        public string version;
    }

    // ----------------------------------------------------------------
    // Story Director models
    // ----------------------------------------------------------------

    /// <summary>
    /// Response from POST /story/reset.
    /// </summary>
    [Serializable]
    public class StoryResetResponse
    {
        public bool ok;
        public string[] born_removed;
        public string[] deceased_restored;
        public string[] manifest_reloaded;
    }

    /// <summary>
    /// Response from POST /story/activity and GET /story/activity.
    /// </summary>
    [Serializable]
    public class ActivityResponse
    {
        public bool ok;
        public string activity;
        public int activity_set_at_tick;
    }

    /// <summary>
    /// Response from POST /story/pause and POST /story/resume.
    /// </summary>
    [Serializable]
    public class PauseResponse
    {
        public bool ok;
        public bool paused;
        public int paused_at_tick;
    }

    /// <summary>
    /// Response from GET /story/pause_state.
    /// </summary>
    [Serializable]
    public class PauseStateResponse
    {
        public bool paused;
        public int paused_at_tick;
        public float tick_budget_seconds;
        public float window_llm_seconds_used;
        public bool budget_exceeded;
        public int next_tick_recommended_in_seconds;
    }

    /// <summary>
    /// Response from POST /story/tick_budget.
    /// </summary>
    [Serializable]
    public class TickBudgetResponse
    {
        public bool ok;
        public float tick_budget_seconds;
    }

    /// <summary>
    /// Response from POST /story/quest_pacing and GET /story/quest_pacing.
    /// </summary>
    [Serializable]
    public class QuestPacingResponse
    {
        public bool ok;
        public int max_unoffered;
        public int cooldown_ticks;
    }

    /// <summary>
    /// Generic response with only an ok field.
    /// Used by endpoints that return {ok: true/false} with no additional data.
    /// </summary>
    [Serializable]
    public class GenericOkResponse
    {
        public bool ok;
    }

    /// <summary>
    /// Response from POST /quests/refuse.
    /// </summary>
    [Serializable]
    public class QuestRefusalResponse
    {
        public bool ok;
        public string quest_id;
        public string npc_id;
        public int trust_delta;
        public string refusal_mode;
    }

    /// <summary>
    /// Response from POST /player/auto_refuse and GET /player/auto_refuse.
    /// </summary>
    [Serializable]
    public class AutoRefuseResponse
    {
        public bool ok;
        public string[] intents;
        public bool dev_enabled;
    }

    /// <summary>
    /// Response from POST /player/introduce.
    /// </summary>
    [Serializable]
    public class IntroduceResponse
    {
        public bool ok;
        public string npc_id;
        public string[] known_as;
        public int max_trust;
    }

    /// <summary>
    /// Response from POST /player/visible_feature.
    /// </summary>
    [Serializable]
    public class VisibleFeatureResponse
    {
        public bool ok;
        public string player_visible_feature;
    }

    /// <summary>
    /// Response from POST /player/register_feature.
    /// </summary>
    [Serializable]
    public class RegisterFeatureResponse
    {
        public bool ok;
        public string feature;
        public string identity;
    }

    /// <summary>
    /// Response from POST /player/vouched_by.
    /// </summary>
    [Serializable]
    public class VouchResponse
    {
        public bool ok;
        public string voucher_npc;
        public string to_npc;
        public string[] known_as;
        public int max_trust;
    }

    /// <summary>
    /// Response from POST /story/tick.
    /// </summary>
    [Serializable]
    public class StoryTickResponse
    {
        public bool ok;
        public int tick;
        public string action_kind;
        public string target_npc;
        public string summary;
        public int next_tick_recommended_in_seconds;
        public bool paused;
    }
}
