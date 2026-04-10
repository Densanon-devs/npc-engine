// Copyright NPC Engine. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "NPCEngineTypes.h"
#include "NPCEngineBlueprintLibrary.generated.h"

/**
 * Static Blueprint helper functions for working with NPC Engine response data.
 */
UCLASS()
class NPCENGINE_API UNPCEngineBlueprintLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()

public:
    /** Extract the dialogue text from a generate response. */
    UFUNCTION(BlueprintPure, Category = "NPC Engine|Helpers",
        meta = (DisplayName = "Get Dialogue"))
    static FString GetDialogue(const FNPCGenerateResponse& Response);

    /** Extract the emotion from a generate response. */
    UFUNCTION(BlueprintPure, Category = "NPC Engine|Helpers",
        meta = (DisplayName = "Get Emotion"))
    static FString GetEmotion(const FNPCGenerateResponse& Response);

    /** Extract the action from a generate response. */
    UFUNCTION(BlueprintPure, Category = "NPC Engine|Helpers",
        meta = (DisplayName = "Get Action"))
    static FString GetAction(const FNPCGenerateResponse& Response);

    /** Check if the generate response includes a quest. */
    UFUNCTION(BlueprintPure, Category = "NPC Engine|Helpers",
        meta = (DisplayName = "Has Quest"))
    static bool HasQuest(const FNPCGenerateResponse& Response);

    /** Extract quest data from a generate response. */
    UFUNCTION(BlueprintPure, Category = "NPC Engine|Helpers",
        meta = (DisplayName = "Get Quest"))
    static FNPCQuestData GetQuest(const FNPCGenerateResponse& Response);

    /** Parse a raw JSON string into dialogue content. Useful for manual parsing. */
    UFUNCTION(BlueprintPure, Category = "NPC Engine|Helpers",
        meta = (DisplayName = "Parse Raw Response"))
    static FNPCDialogueContent ParseRawResponse(const FString& RawJson);
};
