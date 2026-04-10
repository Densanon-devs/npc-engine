using UnityEngine;

namespace NPCEngine
{
    /// <summary>
    /// Handles double-JSON parsing for NPC Engine responses.
    /// The server returns <c>{"response": "{\"dialogue\":...}"}</c> where the
    /// <c>response</c> field is itself a JSON-encoded string. This parser
    /// unwraps that inner JSON into an <see cref="NPCDialogueContent"/> object.
    /// </summary>
    public static class NPCResponseParser
    {
        /// <summary>
        /// Parses a raw JSON string (the inner response value) into dialogue content.
        /// </summary>
        /// <param name="rawJsonString">The JSON string extracted from the response field.</param>
        /// <returns>Parsed <see cref="NPCDialogueContent"/>, or null on failure.</returns>
        public static NPCDialogueContent Parse(string rawJsonString)
        {
            if (string.IsNullOrEmpty(rawJsonString))
            {
                Debug.LogWarning("[NPCEngine] Cannot parse null or empty response string.");
                return null;
            }

            try
            {
                return JsonUtility.FromJson<NPCDialogueContent>(rawJsonString);
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"[NPCEngine] Failed to parse NPC dialogue content: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Attempts to parse a raw JSON string into dialogue content without throwing.
        /// </summary>
        /// <param name="rawJsonString">The JSON string extracted from the response field.</param>
        /// <param name="result">The parsed result, or null if parsing failed.</param>
        /// <returns>True if parsing succeeded, false otherwise.</returns>
        public static bool TryParse(string rawJsonString, out NPCDialogueContent result)
        {
            result = null;

            if (string.IsNullOrEmpty(rawJsonString))
                return false;

            try
            {
                result = JsonUtility.FromJson<NPCDialogueContent>(rawJsonString);
                return result != null;
            }
            catch
            {
                return false;
            }
        }
    }
}
