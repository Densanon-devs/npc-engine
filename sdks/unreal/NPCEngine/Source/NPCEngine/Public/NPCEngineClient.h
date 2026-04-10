// Copyright NPC Engine. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "NPCEngineTypes.h"
#include "NPCEngineClient.generated.h"

// Async callback delegates
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnGenerateResponse, const FNPCGenerateResponse&, Response);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnNPCList, const FNPCListResponse&, Response);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnNPCSwitched, const FNPCInfo&, NpcInfo);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnTrustAdjusted, const FNPCTrustResponse&, Response);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnMoodSet, const FNPCMoodResponse&, Response);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnEventInjected, const FNPCEventResponse&, Response);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnHealthCheck, const FNPCHealthResponse&, Response);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnRequestFailed, const FString&, Endpoint, const FString&, Error);

class IHttpRequest;
class IHttpResponse;

/**
 * HTTP client wrapper for the NPC Engine REST API.
 * Provides Blueprint-callable methods for all NPC Engine endpoints.
 */
UCLASS(BlueprintType, Blueprintable)
class NPCENGINE_API UNPCEngineClient : public UObject
{
    GENERATED_BODY()

public:
    UNPCEngineClient();

    /** Base URL for the NPC Engine server. */
    UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "NPC Engine")
    FString ServerUrl;

    // --- Delegates ---

    /** Fired when a /generate response is received. */
    UPROPERTY(BlueprintAssignable, Category = "NPC Engine|Delegates")
    FOnGenerateResponse OnGenerateResponse;

    /** Fired when the NPC list is received. */
    UPROPERTY(BlueprintAssignable, Category = "NPC Engine|Delegates")
    FOnNPCList OnNPCList;

    /** Fired when the active NPC is switched. */
    UPROPERTY(BlueprintAssignable, Category = "NPC Engine|Delegates")
    FOnNPCSwitched OnNPCSwitched;

    /** Fired when trust is adjusted. */
    UPROPERTY(BlueprintAssignable, Category = "NPC Engine|Delegates")
    FOnTrustAdjusted OnTrustAdjusted;

    /** Fired when mood is set. */
    UPROPERTY(BlueprintAssignable, Category = "NPC Engine|Delegates")
    FOnMoodSet OnMoodSet;

    /** Fired when an event is injected. */
    UPROPERTY(BlueprintAssignable, Category = "NPC Engine|Delegates")
    FOnEventInjected OnEventInjected;

    /** Fired when a health check response is received. */
    UPROPERTY(BlueprintAssignable, Category = "NPC Engine|Delegates")
    FOnHealthCheck OnHealthCheck;

    /** Fired when any request fails. */
    UPROPERTY(BlueprintAssignable, Category = "NPC Engine|Delegates")
    FOnRequestFailed OnRequestFailed;

    // --- API Methods ---

    /** Generate NPC dialogue from a player prompt. */
    UFUNCTION(BlueprintCallable, Category = "NPC Engine")
    void Generate(const FString& Prompt, const FString& NpcId = TEXT(""));

    /** List all available NPCs. */
    UFUNCTION(BlueprintCallable, Category = "NPC Engine")
    void ListNPCs();

    /** Switch the active NPC. */
    UFUNCTION(BlueprintCallable, Category = "NPC Engine")
    void SwitchNPC(const FString& NpcId);

    /** Inject a world event into an NPC's context. */
    UFUNCTION(BlueprintCallable, Category = "NPC Engine")
    void InjectEvent(const FString& Description, const FString& NpcId = TEXT(""));

    /** Adjust trust level for an NPC. */
    UFUNCTION(BlueprintCallable, Category = "NPC Engine")
    void AdjustTrust(const FString& NpcId, int32 Delta, const FString& Reason = TEXT(""));

    /** Set the mood of an NPC. */
    UFUNCTION(BlueprintCallable, Category = "NPC Engine")
    void SetMood(const FString& NpcId, const FString& Mood, float Intensity = 0.5f);

    /** Add a scratchpad entry for an NPC. */
    UFUNCTION(BlueprintCallable, Category = "NPC Engine")
    void AddScratchpad(const FString& NpcId, const FString& Text, float Importance = 0.5f);

    /** Accept a quest from an NPC. */
    UFUNCTION(BlueprintCallable, Category = "NPC Engine")
    void AcceptQuest(const FString& QuestId, const FString& QuestName, const FString& GivenBy);

    /** Mark a quest as completed. */
    UFUNCTION(BlueprintCallable, Category = "NPC Engine")
    void CompleteQuest(const FString& QuestId);

    /** Check if the NPC Engine server is healthy. */
    UFUNCTION(BlueprintCallable, Category = "NPC Engine")
    void CheckHealth();

private:
    /**
     * Send an HTTP request to the NPC Engine API.
     * @param Verb       HTTP method (GET, POST, PUT, DELETE, etc.)
     * @param Path       API path (e.g. "/generate")
     * @param Body       Optional JSON body string; empty for GET requests.
     * @param Callback   Lambda invoked with the response JSON object on success.
     */
    void SendRequest(
        const FString& Verb,
        const FString& Path,
        const FString& Body,
        TFunction<void(TSharedPtr<FJsonObject>)> Callback
    );

    /**
     * Parse the double-encoded dialogue content from a raw JSON string.
     * The API returns a "response" field containing a JSON string that itself
     * contains dialogue, emotion, action, and optional quest data.
     */
    FNPCDialogueContent ParseDialogueContent(const FString& RawJson);
};
