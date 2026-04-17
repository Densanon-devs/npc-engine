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

// Story Director delegates
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnStoryReset, const FStoryResetResponse&, Response);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnActivityResponse, const FActivityResponse&, Response);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnPauseResponse, const FPauseResponse&, Response);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnPauseStateResponse, const FPauseStateResponse&, Response);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnTickBudgetResponse, const FTickBudgetResponse&, Response);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnQuestPacingResponse, const FQuestPacingResponse&, Response);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnRawJsonResponse, const FString&, JsonString);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnQuestRefusalResponse, const FQuestRefusalResponse&, Response);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnAutoRefuseResponse, const FAutoRefuseResponse&, Response);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnIntroduceResponse, const FIntroduceResponse&, Response);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnVisibleFeatureResponse, const FVisibleFeatureResponse&, Response);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnRegisterFeatureResponse, const FRegisterFeatureResponse&, Response);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnVouchResponse, const FVouchResponse&, Response);

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

    // --- Story Director Delegates ---

    /** Fired when a story reset response is received. */
    UPROPERTY(BlueprintAssignable, Category = "NPC Engine|Delegates")
    FOnStoryReset OnStoryReset;

    /** Fired when an activity response is received. */
    UPROPERTY(BlueprintAssignable, Category = "NPC Engine|Delegates")
    FOnActivityResponse OnActivityResponse;

    /** Fired when a pause/resume response is received. */
    UPROPERTY(BlueprintAssignable, Category = "NPC Engine|Delegates")
    FOnPauseResponse OnPauseResponse;

    /** Fired when the pause state is received. */
    UPROPERTY(BlueprintAssignable, Category = "NPC Engine|Delegates")
    FOnPauseStateResponse OnPauseStateResponse;

    /** Fired when the tick budget response is received. */
    UPROPERTY(BlueprintAssignable, Category = "NPC Engine|Delegates")
    FOnTickBudgetResponse OnTickBudgetResponse;

    /** Fired when a quest pacing response is received. */
    UPROPERTY(BlueprintAssignable, Category = "NPC Engine|Delegates")
    FOnQuestPacingResponse OnQuestPacingResponse;

    /** Fired when a raw JSON response is received (graveyard, population, identity_state, reputation). */
    UPROPERTY(BlueprintAssignable, Category = "NPC Engine|Delegates")
    FOnRawJsonResponse OnRawJsonResponse;

    /** Fired when a quest refusal response is received. */
    UPROPERTY(BlueprintAssignable, Category = "NPC Engine|Delegates")
    FOnQuestRefusalResponse OnQuestRefusalResponse;

    /** Fired when an auto-refuse response is received. */
    UPROPERTY(BlueprintAssignable, Category = "NPC Engine|Delegates")
    FOnAutoRefuseResponse OnAutoRefuseResponse;

    /** Fired when a player introduction response is received. */
    UPROPERTY(BlueprintAssignable, Category = "NPC Engine|Delegates")
    FOnIntroduceResponse OnIntroduceResponse;

    /** Fired when a visible feature response is received. */
    UPROPERTY(BlueprintAssignable, Category = "NPC Engine|Delegates")
    FOnVisibleFeatureResponse OnVisibleFeatureResponse;

    /** Fired when a register feature response is received. */
    UPROPERTY(BlueprintAssignable, Category = "NPC Engine|Delegates")
    FOnRegisterFeatureResponse OnRegisterFeatureResponse;

    /** Fired when a vouch response is received. */
    UPROPERTY(BlueprintAssignable, Category = "NPC Engine|Delegates")
    FOnVouchResponse OnVouchResponse;

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
    void SetMood(const FString& NpcId, const FString& Mood, float Intensity = 0.5f, int32 PinTurns = 3);

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

    // --- Story Director Methods ---

    /** Reset the story to its YAML baseline. */
    UFUNCTION(BlueprintCallable, Category = "NPC Engine")
    void StoryReset();

    /** Set the player's current activity context. */
    UFUNCTION(BlueprintCallable, Category = "NPC Engine")
    void SetActivity(const FString& Activity);

    /** Get the player's current activity context. */
    UFUNCTION(BlueprintCallable, Category = "NPC Engine")
    void GetActivity();

    /** Pause all future story ticks. */
    UFUNCTION(BlueprintCallable, Category = "NPC Engine")
    void PauseStory();

    /** Resume story ticks after a pause. */
    UFUNCTION(BlueprintCallable, Category = "NPC Engine")
    void ResumeStory();

    /** Get the current pause state and budget info. */
    UFUNCTION(BlueprintCallable, Category = "NPC Engine")
    void GetPauseState();

    /** Set the rolling-window LLM-time cap (max seconds per minute). */
    UFUNCTION(BlueprintCallable, Category = "NPC Engine")
    void SetTickBudget(float MaxSecondsPerMinute);

    /** Set per-NPC quest pacing overrides. Only fields >= 0 are sent. */
    UFUNCTION(BlueprintCallable, Category = "NPC Engine")
    void SetQuestPacing(int32 MaxUnoffered = -1, int32 CooldownTicks = -1);

    /** Get the current quest pacing configuration. */
    UFUNCTION(BlueprintCallable, Category = "NPC Engine")
    void GetQuestPacing();

    /** Queue a death dispatch for an NPC. */
    UFUNCTION(BlueprintCallable, Category = "NPC Engine")
    void QueueNPCDeath(const FString& NpcId, const FString& Cause = TEXT(""), const FString& TransfersQuestsTo = TEXT(""));

    /** Get the graveyard (deceased NPCs). Broadcasts raw JSON via OnRawJsonResponse. */
    UFUNCTION(BlueprintCallable, Category = "NPC Engine")
    void GetGraveyard();

    /** Queue a birth request for a zone. */
    UFUNCTION(BlueprintCallable, Category = "NPC Engine")
    void QueueNPCBirth(const FString& Zone, const FString& Role = TEXT(""));

    /** Get population stats per zone. Broadcasts raw JSON via OnRawJsonResponse. */
    UFUNCTION(BlueprintCallable, Category = "NPC Engine")
    void GetPopulation();

    /** Refuse a quest from an NPC. */
    UFUNCTION(BlueprintCallable, Category = "NPC Engine")
    void RefuseQuest(const FString& QuestId, const FString& NpcId, const FString& Reason = TEXT(""));

    /** Set the player's auto-refuse intent filter. */
    UFUNCTION(BlueprintCallable, Category = "NPC Engine")
    void SetAutoRefuse(const TArray<FString>& Intents);

    /** Get the current auto-refuse configuration. */
    UFUNCTION(BlueprintCallable, Category = "NPC Engine")
    void GetAutoRefuse();

    /** Introduce the player to an NPC by name and titles. */
    UFUNCTION(BlueprintCallable, Category = "NPC Engine")
    void IntroducePlayer(const FString& ToNpc, const FString& Name, const TArray<FString>& Titles);

    /** Set a player-visible feature (cloak, weapon, etc.). */
    UFUNCTION(BlueprintCallable, Category = "NPC Engine")
    void SetVisibleFeature(const FString& Feature);

    /** Map a visible feature to an identity for auto-recognition. */
    UFUNCTION(BlueprintCallable, Category = "NPC Engine")
    void RegisterFeature(const FString& Feature, const FString& Identity);

    /** Have one NPC vouch the player to another NPC. */
    UFUNCTION(BlueprintCallable, Category = "NPC Engine")
    void VouchPlayer(const FString& VoucherNpc, const FString& ToNpc);

    /** Get per-NPC identity state. Broadcasts raw JSON via OnRawJsonResponse. */
    UFUNCTION(BlueprintCallable, Category = "NPC Engine")
    void GetIdentityState();

    /** Get aggregated reputation data. Broadcasts raw JSON via OnRawJsonResponse. */
    UFUNCTION(BlueprintCallable, Category = "NPC Engine")
    void GetReputation();

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
