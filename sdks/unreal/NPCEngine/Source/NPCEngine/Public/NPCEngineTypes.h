// Copyright NPC Engine. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "NPCEngineTypes.generated.h"

/**
 * Quest data returned when an NPC offers a quest.
 */
USTRUCT(BlueprintType)
struct NPCENGINE_API FNPCQuestData
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Quest")
    FString Type;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Quest")
    FString Objective;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Quest")
    FString Reward;
};

/**
 * Parsed dialogue content from an NPC response.
 */
USTRUCT(BlueprintType)
struct NPCENGINE_API FNPCDialogueContent
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Dialogue")
    FString Dialogue;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Dialogue")
    FString Emotion;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Dialogue")
    FString Action;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Dialogue")
    bool bHasQuest = false;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Dialogue")
    FNPCQuestData Quest;
};

/**
 * Full generation response from the /generate endpoint.
 */
USTRUCT(BlueprintType)
struct NPCENGINE_API FNPCGenerateResponse
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Generate")
    FString NpcId;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Generate")
    FString RawResponse;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Generate")
    FNPCDialogueContent Parsed;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Generate")
    float GenerationTime = 0.0f;
};

/**
 * Information about a single NPC.
 */
USTRUCT(BlueprintType)
struct NPCENGINE_API FNPCInfo
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|NPC")
    FString Id;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|NPC")
    FString Name;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|NPC")
    FString Role;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|NPC")
    TArray<FString> Capabilities;
};

/**
 * Response from /npcs listing all available NPCs.
 */
USTRUCT(BlueprintType)
struct NPCENGINE_API FNPCListResponse
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|NPC")
    FString ActiveNpc;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|NPC")
    FString WorldName;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|NPC")
    TArray<FNPCInfo> Npcs;
};

/**
 * Response from trust adjustment.
 */
USTRUCT(BlueprintType)
struct NPCENGINE_API FNPCTrustResponse
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Trust")
    FString NpcId;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Trust")
    int32 OldLevel = 0;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Trust")
    int32 NewLevel = 0;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Trust")
    int32 Delta = 0;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Trust")
    FString Reason;
};

/**
 * Response from mood adjustment.
 */
USTRUCT(BlueprintType)
struct NPCENGINE_API FNPCMoodResponse
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Mood")
    FString NpcId;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Mood")
    FString OldMood;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Mood")
    FString NewMood;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Mood")
    float Intensity = 0.0f;
};

/**
 * Response from event injection.
 */
USTRUCT(BlueprintType)
struct NPCENGINE_API FNPCEventResponse
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Event")
    bool bInjected = false;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Event")
    FString Target;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Event")
    FString EventDescription;
};

/**
 * Response from health check endpoint.
 */
USTRUCT(BlueprintType)
struct NPCENGINE_API FNPCHealthResponse
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Health")
    FString Status;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Health")
    FString Version;
};

// ---------------------------------------------------------------------------
// Story Director types
// ---------------------------------------------------------------------------

/**
 * Response from POST /story/reset.
 */
USTRUCT(BlueprintType)
struct NPCENGINE_API FStoryResetResponse
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Story")
    bool bOk = false;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Story")
    TArray<FString> BornRemoved;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Story")
    TArray<FString> DeceasedRestored;
};

/**
 * Response from POST/GET /story/activity.
 */
USTRUCT(BlueprintType)
struct NPCENGINE_API FActivityResponse
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Story")
    bool bOk = false;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Story")
    FString Activity;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Story")
    int32 ActivitySetAtTick = 0;
};

/**
 * Response from POST /story/pause and POST /story/resume.
 */
USTRUCT(BlueprintType)
struct NPCENGINE_API FPauseResponse
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Story")
    bool bOk = false;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Story")
    bool bPaused = false;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Story")
    int32 PausedAtTick = 0;
};

/**
 * Response from GET /story/pause_state.
 */
USTRUCT(BlueprintType)
struct NPCENGINE_API FPauseStateResponse
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Story")
    bool bPaused = false;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Story")
    int32 PausedAtTick = 0;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Story")
    float TickBudgetSeconds = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Story")
    float WindowLLMSecondsUsed = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Story")
    bool bBudgetExceeded = false;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Story")
    float NextTickRecommendedInSeconds = 0.0f;
};

/**
 * Response from POST /story/tick_budget.
 */
USTRUCT(BlueprintType)
struct NPCENGINE_API FTickBudgetResponse
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Story")
    bool bOk = false;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Story")
    float TickBudgetSeconds = 0.0f;
};

/**
 * Response from POST/GET /story/quest_pacing.
 */
USTRUCT(BlueprintType)
struct NPCENGINE_API FQuestPacingResponse
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Story")
    bool bOk = false;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Story")
    int32 MaxUnoffered = 0;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Story")
    int32 CooldownTicks = 0;
};

/**
 * Response from POST /quests/refuse.
 */
USTRUCT(BlueprintType)
struct NPCENGINE_API FQuestRefusalResponse
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Quest")
    bool bOk = false;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Quest")
    FString QuestId;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Quest")
    FString NpcId;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Quest")
    int32 TrustDelta = 0;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Quest")
    FString RefusalMode;
};

/**
 * Response from POST/GET /player/auto_refuse.
 */
USTRUCT(BlueprintType)
struct NPCENGINE_API FAutoRefuseResponse
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Player")
    bool bOk = false;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Player")
    TArray<FString> Intents;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Player")
    bool bDevEnabled = false;
};

/**
 * Response from POST /player/introduce.
 */
USTRUCT(BlueprintType)
struct NPCENGINE_API FIntroduceResponse
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Player")
    bool bOk = false;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Player")
    FString NpcId;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Player")
    TArray<FString> KnownAs;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Player")
    int32 MaxTrust = 0;
};

/**
 * Response from POST /player/visible_feature.
 */
USTRUCT(BlueprintType)
struct NPCENGINE_API FVisibleFeatureResponse
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Player")
    bool bOk = false;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Player")
    FString PlayerVisibleFeature;
};

/**
 * Response from POST /player/register_feature.
 */
USTRUCT(BlueprintType)
struct NPCENGINE_API FRegisterFeatureResponse
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Player")
    bool bOk = false;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Player")
    FString Feature;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Player")
    FString Identity;
};

/**
 * Response from POST /player/vouched_by.
 */
USTRUCT(BlueprintType)
struct NPCENGINE_API FVouchResponse
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Player")
    bool bOk = false;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Player")
    FString VoucherNpc;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Player")
    FString ToNpc;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Player")
    TArray<FString> KnownAs;

    UPROPERTY(BlueprintReadOnly, Category = "NPC Engine|Player")
    int32 MaxTrust = 0;
};
