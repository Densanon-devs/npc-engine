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
