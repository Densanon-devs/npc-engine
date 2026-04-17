// Copyright NPC Engine. All Rights Reserved.

#include "NPCEngineClient.h"
#include "HttpModule.h"
#include "Interfaces/IHttpRequest.h"
#include "Interfaces/IHttpResponse.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"

UNPCEngineClient::UNPCEngineClient()
    : ServerUrl(TEXT("http://127.0.0.1:8000"))
{
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

void UNPCEngineClient::SendRequest(
    const FString& Verb,
    const FString& Path,
    const FString& Body,
    TFunction<void(TSharedPtr<FJsonObject>)> Callback)
{
    const FString Url = ServerUrl + Path;

    TSharedRef<IHttpRequest, ESPMode::ThreadSafe> Request = FHttpModule::Get().CreateRequest();
    Request->SetURL(Url);
    Request->SetVerb(Verb);
    Request->SetHeader(TEXT("Content-Type"), TEXT("application/json"));
    Request->SetHeader(TEXT("Accept"), TEXT("application/json"));

    if (!Body.IsEmpty())
    {
        Request->SetContentAsString(Body);
    }

    // Prevent garbage collection of this UObject while the request is in flight.
    AddToRoot();

    // Capture a weak pointer so we can check validity in the async callback.
    TWeakObjectPtr<UNPCEngineClient> WeakThis(this);
    FString CapturedPath = Path;

    Request->OnProcessRequestComplete().BindLambda(
        [WeakThis, Callback, CapturedPath](FHttpRequestPtr /*Req*/, FHttpResponsePtr Resp, bool bConnectedSuccessfully)
        {
            UNPCEngineClient* Self = WeakThis.Get();
            if (!Self)
            {
                return;
            }

            // Safe to remove from root now that the request has completed.
            Self->RemoveFromRoot();

            if (!bConnectedSuccessfully || !Resp.IsValid())
            {
                Self->OnRequestFailed.Broadcast(CapturedPath, TEXT("Connection failed"));
                return;
            }

            const int32 Code = Resp->GetResponseCode();
            const FString ResponseBody = Resp->GetContentAsString();

            if (Code < 200 || Code >= 300)
            {
                Self->OnRequestFailed.Broadcast(
                    CapturedPath,
                    FString::Printf(TEXT("HTTP %d: %s"), Code, *ResponseBody));
                return;
            }

            TSharedPtr<FJsonObject> JsonObject;
            TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(ResponseBody);

            if (!FJsonSerializer::Deserialize(Reader, JsonObject) || !JsonObject.IsValid())
            {
                Self->OnRequestFailed.Broadcast(CapturedPath, TEXT("Failed to parse JSON response"));
                return;
            }

            Callback(JsonObject);
        });

    Request->ProcessRequest();
}

static FString JsonObjectToString(const TSharedRef<FJsonObject>& Obj)
{
    FString Output;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Output);
    FJsonSerializer::Serialize(Obj, Writer);
    return Output;
}

FNPCDialogueContent UNPCEngineClient::ParseDialogueContent(const FString& RawJson)
{
    FNPCDialogueContent Content;

    TSharedPtr<FJsonObject> JsonObject;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(RawJson);

    if (!FJsonSerializer::Deserialize(Reader, JsonObject) || !JsonObject.IsValid())
    {
        // If it's not valid JSON, treat the whole thing as dialogue text.
        Content.Dialogue = RawJson;
        return Content;
    }

    Content.Dialogue = JsonObject->GetStringField(TEXT("dialogue"));
    Content.Emotion = JsonObject->GetStringField(TEXT("emotion"));
    Content.Action = JsonObject->GetStringField(TEXT("action"));

    // Quest data is optional.
    const TSharedPtr<FJsonObject>* QuestObj = nullptr;
    if (JsonObject->TryGetObjectField(TEXT("quest"), QuestObj) && QuestObj && (*QuestObj).IsValid())
    {
        Content.bHasQuest = true;
        Content.Quest.Type = (*QuestObj)->GetStringField(TEXT("type"));
        Content.Quest.Objective = (*QuestObj)->GetStringField(TEXT("objective"));
        Content.Quest.Reward = (*QuestObj)->GetStringField(TEXT("reward"));
    }

    return Content;
}

// ---------------------------------------------------------------------------
// API methods
// ---------------------------------------------------------------------------

void UNPCEngineClient::Generate(const FString& Prompt, const FString& NpcId)
{
    TSharedRef<FJsonObject> Body = MakeShared<FJsonObject>();
    Body->SetStringField(TEXT("prompt"), Prompt);

    if (!NpcId.IsEmpty())
    {
        Body->SetStringField(TEXT("npc_id"), NpcId);
    }

    SendRequest(TEXT("POST"), TEXT("/generate"), JsonObjectToString(Body),
        [this](TSharedPtr<FJsonObject> Json)
        {
            FNPCGenerateResponse Response;
            Response.NpcId = Json->GetStringField(TEXT("npc_id"));
            Response.GenerationTime = static_cast<float>(Json->GetNumberField(TEXT("generation_time")));

            // The "response" field may be a JSON string (double-encoded) or plain text.
            Response.RawResponse = Json->GetStringField(TEXT("response"));
            Response.Parsed = ParseDialogueContent(Response.RawResponse);

            OnGenerateResponse.Broadcast(Response);
        });
}

void UNPCEngineClient::ListNPCs()
{
    SendRequest(TEXT("GET"), TEXT("/npc/list"), TEXT(""),
        [this](TSharedPtr<FJsonObject> Json)
        {
            FNPCListResponse Response;
            Response.ActiveNpc = Json->GetStringField(TEXT("active_npc"));
            Response.WorldName = Json->GetStringField(TEXT("world_name"));

            const TArray<TSharedPtr<FJsonValue>>* NpcArray = nullptr;
            if (Json->TryGetArrayField(TEXT("npcs"), NpcArray) && NpcArray)
            {
                for (const TSharedPtr<FJsonValue>& Val : *NpcArray)
                {
                    const TSharedPtr<FJsonObject>& Obj = Val->AsObject();
                    if (!Obj.IsValid()) continue;

                    FNPCInfo Info;
                    Info.Id = Obj->GetStringField(TEXT("id"));
                    Info.Name = Obj->GetStringField(TEXT("name"));
                    Info.Role = Obj->GetStringField(TEXT("role"));

                    const TArray<TSharedPtr<FJsonValue>>* CapsArray = nullptr;
                    if (Obj->TryGetArrayField(TEXT("capabilities"), CapsArray) && CapsArray)
                    {
                        for (const TSharedPtr<FJsonValue>& Cap : *CapsArray)
                        {
                            Info.Capabilities.Add(Cap->AsString());
                        }
                    }

                    Response.Npcs.Add(Info);
                }
            }

            OnNPCList.Broadcast(Response);
        });
}

void UNPCEngineClient::SwitchNPC(const FString& NpcId)
{
    TSharedRef<FJsonObject> Body = MakeShared<FJsonObject>();
    Body->SetStringField(TEXT("npc_id"), NpcId);

    SendRequest(TEXT("POST"), TEXT("/npc/switch"), JsonObjectToString(Body),
        [this](TSharedPtr<FJsonObject> Json)
        {
            FNPCInfo Info;
            Info.Id = Json->GetStringField(TEXT("id"));
            Info.Name = Json->GetStringField(TEXT("name"));
            Info.Role = Json->GetStringField(TEXT("role"));

            const TArray<TSharedPtr<FJsonValue>>* CapsArray = nullptr;
            if (Json->TryGetArrayField(TEXT("capabilities"), CapsArray) && CapsArray)
            {
                for (const TSharedPtr<FJsonValue>& Cap : *CapsArray)
                {
                    Info.Capabilities.Add(Cap->AsString());
                }
            }

            OnNPCSwitched.Broadcast(Info);
        });
}

void UNPCEngineClient::InjectEvent(const FString& Description, const FString& NpcId)
{
    TSharedRef<FJsonObject> Body = MakeShared<FJsonObject>();
    Body->SetStringField(TEXT("description"), Description);

    if (!NpcId.IsEmpty())
    {
        Body->SetStringField(TEXT("npc_id"), NpcId);
    }

    SendRequest(TEXT("POST"), TEXT("/events/inject"), JsonObjectToString(Body),
        [this](TSharedPtr<FJsonObject> Json)
        {
            FNPCEventResponse Response;
            Response.bInjected = Json->GetBoolField(TEXT("injected"));
            Response.Target = Json->GetStringField(TEXT("target"));
            Response.EventDescription = Json->GetStringField(TEXT("description"));

            OnEventInjected.Broadcast(Response);
        });
}

void UNPCEngineClient::AdjustTrust(const FString& NpcId, int32 Delta, const FString& Reason)
{
    TSharedRef<FJsonObject> Body = MakeShared<FJsonObject>();
    Body->SetStringField(TEXT("npc_id"), NpcId);
    Body->SetNumberField(TEXT("delta"), Delta);

    if (!Reason.IsEmpty())
    {
        Body->SetStringField(TEXT("reason"), Reason);
    }

    SendRequest(TEXT("POST"), TEXT("/npc/trust"), JsonObjectToString(Body),
        [this](TSharedPtr<FJsonObject> Json)
        {
            FNPCTrustResponse Response;
            Response.NpcId = Json->GetStringField(TEXT("npc_id"));
            Response.OldLevel = static_cast<int32>(Json->GetNumberField(TEXT("old_level")));
            Response.NewLevel = static_cast<int32>(Json->GetNumberField(TEXT("new_level")));
            Response.Delta = static_cast<int32>(Json->GetNumberField(TEXT("delta")));
            Response.Reason = Json->GetStringField(TEXT("reason"));

            OnTrustAdjusted.Broadcast(Response);
        });
}

void UNPCEngineClient::SetMood(const FString& NpcId, const FString& Mood, float Intensity, int32 PinTurns)
{
    TSharedRef<FJsonObject> Body = MakeShared<FJsonObject>();
    Body->SetStringField(TEXT("npc_id"), NpcId);
    Body->SetStringField(TEXT("mood"), Mood);
    Body->SetNumberField(TEXT("intensity"), Intensity);
    Body->SetNumberField(TEXT("pin_turns"), PinTurns);

    SendRequest(TEXT("POST"), TEXT("/npc/mood"), JsonObjectToString(Body),
        [this](TSharedPtr<FJsonObject> Json)
        {
            FNPCMoodResponse Response;
            Response.NpcId = Json->GetStringField(TEXT("npc_id"));
            Response.OldMood = Json->GetStringField(TEXT("old_mood"));
            Response.NewMood = Json->GetStringField(TEXT("new_mood"));
            Response.Intensity = static_cast<float>(Json->GetNumberField(TEXT("intensity")));

            OnMoodSet.Broadcast(Response);
        });
}

void UNPCEngineClient::AddScratchpad(const FString& NpcId, const FString& Text, float Importance)
{
    TSharedRef<FJsonObject> Body = MakeShared<FJsonObject>();
    Body->SetStringField(TEXT("npc_id"), NpcId);
    Body->SetStringField(TEXT("text"), Text);
    Body->SetNumberField(TEXT("importance"), Importance);

    // Fire-and-forget; no dedicated delegate for scratchpad.
    SendRequest(TEXT("POST"), TEXT("/npc/scratchpad"), JsonObjectToString(Body),
        [](TSharedPtr<FJsonObject> /*Json*/)
        {
            // Success — no broadcast needed.
        });
}

void UNPCEngineClient::AcceptQuest(const FString& QuestId, const FString& QuestName, const FString& GivenBy)
{
    TSharedRef<FJsonObject> Body = MakeShared<FJsonObject>();
    Body->SetStringField(TEXT("quest_id"), QuestId);
    Body->SetStringField(TEXT("quest_name"), QuestName);
    Body->SetStringField(TEXT("given_by"), GivenBy);

    SendRequest(TEXT("POST"), TEXT("/quests/accept"), JsonObjectToString(Body),
        [](TSharedPtr<FJsonObject> /*Json*/)
        {
            // Success — no broadcast needed.
        });
}

void UNPCEngineClient::CompleteQuest(const FString& QuestId)
{
    TSharedRef<FJsonObject> Body = MakeShared<FJsonObject>();
    Body->SetStringField(TEXT("quest_id"), QuestId);

    SendRequest(TEXT("POST"), TEXT("/quests/complete"), JsonObjectToString(Body),
        [](TSharedPtr<FJsonObject> /*Json*/)
        {
            // Success — no broadcast needed.
        });
}

void UNPCEngineClient::CheckHealth()
{
    SendRequest(TEXT("GET"), TEXT("/health"), TEXT(""),
        [this](TSharedPtr<FJsonObject> Json)
        {
            FNPCHealthResponse Response;
            Response.Status = Json->GetStringField(TEXT("status"));
            Response.Version = Json->GetStringField(TEXT("version"));

            OnHealthCheck.Broadcast(Response);
        });
}

// ---------------------------------------------------------------------------
// Story Director methods
// ---------------------------------------------------------------------------

void UNPCEngineClient::StoryReset()
{
    SendRequest(TEXT("POST"), TEXT("/story/reset"), TEXT(""),
        [this](TSharedPtr<FJsonObject> Json)
        {
            FStoryResetResponse Response;
            Response.bOk = Json->GetBoolField(TEXT("ok"));

            const TArray<TSharedPtr<FJsonValue>>* BornArray = nullptr;
            if (Json->TryGetArrayField(TEXT("born_removed"), BornArray) && BornArray)
            {
                for (const TSharedPtr<FJsonValue>& Val : *BornArray)
                {
                    Response.BornRemoved.Add(Val->AsString());
                }
            }

            const TArray<TSharedPtr<FJsonValue>>* DeceasedArray = nullptr;
            if (Json->TryGetArrayField(TEXT("deceased_restored"), DeceasedArray) && DeceasedArray)
            {
                for (const TSharedPtr<FJsonValue>& Val : *DeceasedArray)
                {
                    Response.DeceasedRestored.Add(Val->AsString());
                }
            }

            OnStoryReset.Broadcast(Response);
        });
}

void UNPCEngineClient::SetActivity(const FString& Activity)
{
    TSharedRef<FJsonObject> Body = MakeShared<FJsonObject>();
    Body->SetStringField(TEXT("activity"), Activity);

    SendRequest(TEXT("POST"), TEXT("/story/activity"), JsonObjectToString(Body),
        [this](TSharedPtr<FJsonObject> Json)
        {
            FActivityResponse Response;
            Response.bOk = Json->GetBoolField(TEXT("ok"));
            Response.Activity = Json->GetStringField(TEXT("activity"));
            Response.ActivitySetAtTick = static_cast<int32>(Json->GetNumberField(TEXT("activity_set_at_tick")));

            OnActivityResponse.Broadcast(Response);
        });
}

void UNPCEngineClient::GetActivity()
{
    SendRequest(TEXT("GET"), TEXT("/story/activity"), TEXT(""),
        [this](TSharedPtr<FJsonObject> Json)
        {
            FActivityResponse Response;
            Response.bOk = true;
            Response.Activity = Json->GetStringField(TEXT("activity"));
            Response.ActivitySetAtTick = static_cast<int32>(Json->GetNumberField(TEXT("activity_set_at_tick")));

            OnActivityResponse.Broadcast(Response);
        });
}

void UNPCEngineClient::PauseStory()
{
    SendRequest(TEXT("POST"), TEXT("/story/pause"), TEXT(""),
        [this](TSharedPtr<FJsonObject> Json)
        {
            FPauseResponse Response;
            Response.bOk = Json->GetBoolField(TEXT("ok"));
            Response.bPaused = Json->GetBoolField(TEXT("paused"));
            Response.PausedAtTick = static_cast<int32>(Json->GetNumberField(TEXT("paused_at_tick")));

            OnPauseResponse.Broadcast(Response);
        });
}

void UNPCEngineClient::ResumeStory()
{
    SendRequest(TEXT("POST"), TEXT("/story/resume"), TEXT(""),
        [this](TSharedPtr<FJsonObject> Json)
        {
            FPauseResponse Response;
            Response.bOk = Json->GetBoolField(TEXT("ok"));
            Response.bPaused = Json->GetBoolField(TEXT("paused"));
            Response.PausedAtTick = static_cast<int32>(Json->GetNumberField(TEXT("paused_at_tick")));

            OnPauseResponse.Broadcast(Response);
        });
}

void UNPCEngineClient::GetPauseState()
{
    SendRequest(TEXT("GET"), TEXT("/story/pause_state"), TEXT(""),
        [this](TSharedPtr<FJsonObject> Json)
        {
            FPauseStateResponse Response;
            Response.bPaused = Json->GetBoolField(TEXT("paused"));
            Response.PausedAtTick = static_cast<int32>(Json->GetNumberField(TEXT("paused_at_tick")));
            Response.TickBudgetSeconds = static_cast<float>(Json->GetNumberField(TEXT("tick_budget_seconds")));
            Response.WindowLLMSecondsUsed = static_cast<float>(Json->GetNumberField(TEXT("window_llm_seconds_used")));
            Response.bBudgetExceeded = Json->GetBoolField(TEXT("budget_exceeded"));
            Response.NextTickRecommendedInSeconds = static_cast<float>(Json->GetNumberField(TEXT("next_tick_recommended_in_seconds")));

            OnPauseStateResponse.Broadcast(Response);
        });
}

void UNPCEngineClient::SetTickBudget(float MaxSecondsPerMinute)
{
    TSharedRef<FJsonObject> Body = MakeShared<FJsonObject>();
    Body->SetNumberField(TEXT("max_seconds_per_minute"), MaxSecondsPerMinute);

    SendRequest(TEXT("POST"), TEXT("/story/tick_budget"), JsonObjectToString(Body),
        [this](TSharedPtr<FJsonObject> Json)
        {
            FTickBudgetResponse Response;
            Response.bOk = Json->GetBoolField(TEXT("ok"));
            Response.TickBudgetSeconds = static_cast<float>(Json->GetNumberField(TEXT("tick_budget_seconds")));

            OnTickBudgetResponse.Broadcast(Response);
        });
}

void UNPCEngineClient::SetQuestPacing(int32 MaxUnoffered, int32 CooldownTicks)
{
    TSharedRef<FJsonObject> Body = MakeShared<FJsonObject>();

    if (MaxUnoffered >= 0)
    {
        Body->SetNumberField(TEXT("max_unoffered"), MaxUnoffered);
    }

    if (CooldownTicks >= 0)
    {
        Body->SetNumberField(TEXT("cooldown_ticks"), CooldownTicks);
    }

    SendRequest(TEXT("POST"), TEXT("/story/quest_pacing"), JsonObjectToString(Body),
        [this](TSharedPtr<FJsonObject> Json)
        {
            FQuestPacingResponse Response;
            Response.bOk = Json->GetBoolField(TEXT("ok"));
            Response.MaxUnoffered = static_cast<int32>(Json->GetNumberField(TEXT("max_unoffered")));
            Response.CooldownTicks = static_cast<int32>(Json->GetNumberField(TEXT("cooldown_ticks")));

            OnQuestPacingResponse.Broadcast(Response);
        });
}

void UNPCEngineClient::GetQuestPacing()
{
    SendRequest(TEXT("GET"), TEXT("/story/quest_pacing"), TEXT(""),
        [this](TSharedPtr<FJsonObject> Json)
        {
            FQuestPacingResponse Response;
            Response.bOk = true;
            Response.MaxUnoffered = static_cast<int32>(Json->GetNumberField(TEXT("max_unoffered")));
            Response.CooldownTicks = static_cast<int32>(Json->GetNumberField(TEXT("cooldown_ticks")));

            OnQuestPacingResponse.Broadcast(Response);
        });
}

void UNPCEngineClient::QueueNPCDeath(const FString& NpcId, const FString& Cause, const FString& TransfersQuestsTo)
{
    TSharedRef<FJsonObject> Body = MakeShared<FJsonObject>();
    Body->SetStringField(TEXT("npc_id"), NpcId);

    if (!Cause.IsEmpty())
    {
        Body->SetStringField(TEXT("cause"), Cause);
    }

    if (!TransfersQuestsTo.IsEmpty())
    {
        Body->SetStringField(TEXT("transfers_quests_to"), TransfersQuestsTo);
    }

    SendRequest(TEXT("POST"), TEXT("/story/npc_death"), JsonObjectToString(Body),
        [this](TSharedPtr<FJsonObject> Json)
        {
            FString RawJson;
            TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&RawJson);
            FJsonSerializer::Serialize(Json.ToSharedRef(), Writer);

            OnRawJsonResponse.Broadcast(RawJson);
        });
}

void UNPCEngineClient::GetGraveyard()
{
    SendRequest(TEXT("GET"), TEXT("/story/graveyard"), TEXT(""),
        [this](TSharedPtr<FJsonObject> Json)
        {
            FString RawJson;
            TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&RawJson);
            FJsonSerializer::Serialize(Json.ToSharedRef(), Writer);

            OnRawJsonResponse.Broadcast(RawJson);
        });
}

void UNPCEngineClient::QueueNPCBirth(const FString& Zone, const FString& Role)
{
    TSharedRef<FJsonObject> Body = MakeShared<FJsonObject>();
    Body->SetStringField(TEXT("zone"), Zone);

    if (!Role.IsEmpty())
    {
        Body->SetStringField(TEXT("role"), Role);
    }

    SendRequest(TEXT("POST"), TEXT("/story/npc_birth_request"), JsonObjectToString(Body),
        [this](TSharedPtr<FJsonObject> Json)
        {
            FString RawJson;
            TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&RawJson);
            FJsonSerializer::Serialize(Json.ToSharedRef(), Writer);

            OnRawJsonResponse.Broadcast(RawJson);
        });
}

void UNPCEngineClient::GetPopulation()
{
    SendRequest(TEXT("GET"), TEXT("/story/population"), TEXT(""),
        [this](TSharedPtr<FJsonObject> Json)
        {
            FString RawJson;
            TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&RawJson);
            FJsonSerializer::Serialize(Json.ToSharedRef(), Writer);

            OnRawJsonResponse.Broadcast(RawJson);
        });
}

void UNPCEngineClient::RefuseQuest(const FString& QuestId, const FString& NpcId, const FString& Reason)
{
    TSharedRef<FJsonObject> Body = MakeShared<FJsonObject>();
    Body->SetStringField(TEXT("quest_id"), QuestId);
    Body->SetStringField(TEXT("npc_id"), NpcId);

    if (!Reason.IsEmpty())
    {
        Body->SetStringField(TEXT("reason"), Reason);
    }

    SendRequest(TEXT("POST"), TEXT("/quests/refuse"), JsonObjectToString(Body),
        [this](TSharedPtr<FJsonObject> Json)
        {
            FQuestRefusalResponse Response;
            Response.bOk = Json->GetBoolField(TEXT("ok"));
            Response.QuestId = Json->GetStringField(TEXT("quest_id"));
            Response.NpcId = Json->GetStringField(TEXT("npc_id"));
            Response.TrustDelta = static_cast<int32>(Json->GetNumberField(TEXT("trust_delta")));
            Response.RefusalMode = Json->GetStringField(TEXT("refusal_mode"));

            OnQuestRefusalResponse.Broadcast(Response);
        });
}

void UNPCEngineClient::SetAutoRefuse(const TArray<FString>& Intents)
{
    TSharedRef<FJsonObject> Body = MakeShared<FJsonObject>();

    TArray<TSharedPtr<FJsonValue>> IntentValues;
    for (const FString& Intent : Intents)
    {
        IntentValues.Add(MakeShared<FJsonValueString>(Intent));
    }
    Body->SetArrayField(TEXT("intents"), IntentValues);

    SendRequest(TEXT("POST"), TEXT("/player/auto_refuse"), JsonObjectToString(Body),
        [this](TSharedPtr<FJsonObject> Json)
        {
            FAutoRefuseResponse Response;
            Response.bOk = Json->GetBoolField(TEXT("ok"));
            Response.bDevEnabled = Json->GetBoolField(TEXT("dev_enabled"));

            const TArray<TSharedPtr<FJsonValue>>* IntentArray = nullptr;
            if (Json->TryGetArrayField(TEXT("intents"), IntentArray) && IntentArray)
            {
                for (const TSharedPtr<FJsonValue>& Val : *IntentArray)
                {
                    Response.Intents.Add(Val->AsString());
                }
            }

            OnAutoRefuseResponse.Broadcast(Response);
        });
}

void UNPCEngineClient::GetAutoRefuse()
{
    SendRequest(TEXT("GET"), TEXT("/player/auto_refuse"), TEXT(""),
        [this](TSharedPtr<FJsonObject> Json)
        {
            FAutoRefuseResponse Response;
            Response.bOk = true;
            Response.bDevEnabled = Json->GetBoolField(TEXT("dev_enabled"));

            const TArray<TSharedPtr<FJsonValue>>* IntentArray = nullptr;
            if (Json->TryGetArrayField(TEXT("intents"), IntentArray) && IntentArray)
            {
                for (const TSharedPtr<FJsonValue>& Val : *IntentArray)
                {
                    Response.Intents.Add(Val->AsString());
                }
            }

            OnAutoRefuseResponse.Broadcast(Response);
        });
}

void UNPCEngineClient::IntroducePlayer(const FString& ToNpc, const FString& Name, const TArray<FString>& Titles)
{
    TSharedRef<FJsonObject> Body = MakeShared<FJsonObject>();
    Body->SetStringField(TEXT("to_npc"), ToNpc);
    Body->SetStringField(TEXT("name"), Name);

    TArray<TSharedPtr<FJsonValue>> TitleValues;
    for (const FString& Title : Titles)
    {
        TitleValues.Add(MakeShared<FJsonValueString>(Title));
    }
    Body->SetArrayField(TEXT("titles"), TitleValues);

    SendRequest(TEXT("POST"), TEXT("/player/introduce"), JsonObjectToString(Body),
        [this](TSharedPtr<FJsonObject> Json)
        {
            FIntroduceResponse Response;
            Response.bOk = Json->GetBoolField(TEXT("ok"));
            Response.NpcId = Json->GetStringField(TEXT("npc_id"));
            Response.MaxTrust = static_cast<int32>(Json->GetNumberField(TEXT("max_trust")));

            const TArray<TSharedPtr<FJsonValue>>* KnownArray = nullptr;
            if (Json->TryGetArrayField(TEXT("known_as"), KnownArray) && KnownArray)
            {
                for (const TSharedPtr<FJsonValue>& Val : *KnownArray)
                {
                    Response.KnownAs.Add(Val->AsString());
                }
            }

            OnIntroduceResponse.Broadcast(Response);
        });
}

void UNPCEngineClient::SetVisibleFeature(const FString& Feature)
{
    TSharedRef<FJsonObject> Body = MakeShared<FJsonObject>();
    Body->SetStringField(TEXT("feature"), Feature);

    SendRequest(TEXT("POST"), TEXT("/player/visible_feature"), JsonObjectToString(Body),
        [this](TSharedPtr<FJsonObject> Json)
        {
            FVisibleFeatureResponse Response;
            Response.bOk = Json->GetBoolField(TEXT("ok"));
            Response.PlayerVisibleFeature = Json->GetStringField(TEXT("player_visible_feature"));

            OnVisibleFeatureResponse.Broadcast(Response);
        });
}

void UNPCEngineClient::RegisterFeature(const FString& Feature, const FString& Identity)
{
    TSharedRef<FJsonObject> Body = MakeShared<FJsonObject>();
    Body->SetStringField(TEXT("feature"), Feature);
    Body->SetStringField(TEXT("identity"), Identity);

    SendRequest(TEXT("POST"), TEXT("/player/register_feature"), JsonObjectToString(Body),
        [this](TSharedPtr<FJsonObject> Json)
        {
            FRegisterFeatureResponse Response;
            Response.bOk = Json->GetBoolField(TEXT("ok"));
            Response.Feature = Json->GetStringField(TEXT("feature"));
            Response.Identity = Json->GetStringField(TEXT("identity"));

            OnRegisterFeatureResponse.Broadcast(Response);
        });
}

void UNPCEngineClient::VouchPlayer(const FString& VoucherNpc, const FString& ToNpc)
{
    TSharedRef<FJsonObject> Body = MakeShared<FJsonObject>();
    Body->SetStringField(TEXT("voucher_npc"), VoucherNpc);
    Body->SetStringField(TEXT("to_npc"), ToNpc);

    SendRequest(TEXT("POST"), TEXT("/player/vouched_by"), JsonObjectToString(Body),
        [this](TSharedPtr<FJsonObject> Json)
        {
            FVouchResponse Response;
            Response.bOk = Json->GetBoolField(TEXT("ok"));
            Response.VoucherNpc = Json->GetStringField(TEXT("voucher_npc"));
            Response.ToNpc = Json->GetStringField(TEXT("to_npc"));
            Response.MaxTrust = static_cast<int32>(Json->GetNumberField(TEXT("max_trust")));

            const TArray<TSharedPtr<FJsonValue>>* KnownArray = nullptr;
            if (Json->TryGetArrayField(TEXT("known_as"), KnownArray) && KnownArray)
            {
                for (const TSharedPtr<FJsonValue>& Val : *KnownArray)
                {
                    Response.KnownAs.Add(Val->AsString());
                }
            }

            OnVouchResponse.Broadcast(Response);
        });
}

void UNPCEngineClient::GetIdentityState()
{
    SendRequest(TEXT("GET"), TEXT("/player/identity_state"), TEXT(""),
        [this](TSharedPtr<FJsonObject> Json)
        {
            FString RawJson;
            TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&RawJson);
            FJsonSerializer::Serialize(Json.ToSharedRef(), Writer);

            OnRawJsonResponse.Broadcast(RawJson);
        });
}

void UNPCEngineClient::GetReputation()
{
    SendRequest(TEXT("GET"), TEXT("/player/reputation"), TEXT(""),
        [this](TSharedPtr<FJsonObject> Json)
        {
            FString RawJson;
            TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&RawJson);
            FJsonSerializer::Serialize(Json.ToSharedRef(), Writer);

            OnRawJsonResponse.Broadcast(RawJson);
        });
}
