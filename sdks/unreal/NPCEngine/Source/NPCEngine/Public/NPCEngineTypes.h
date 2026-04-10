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
