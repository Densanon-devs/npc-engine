// Copyright NPC Engine. All Rights Reserved.

#include "NPCEngineBlueprintLibrary.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Dom/JsonObject.h"

FString UNPCEngineBlueprintLibrary::GetDialogue(const FNPCGenerateResponse& Response)
{
    return Response.Parsed.Dialogue;
}

FString UNPCEngineBlueprintLibrary::GetEmotion(const FNPCGenerateResponse& Response)
{
    return Response.Parsed.Emotion;
}

FString UNPCEngineBlueprintLibrary::GetAction(const FNPCGenerateResponse& Response)
{
    return Response.Parsed.Action;
}

bool UNPCEngineBlueprintLibrary::HasQuest(const FNPCGenerateResponse& Response)
{
    return Response.Parsed.bHasQuest;
}

FNPCQuestData UNPCEngineBlueprintLibrary::GetQuest(const FNPCGenerateResponse& Response)
{
    return Response.Parsed.Quest;
}

FNPCDialogueContent UNPCEngineBlueprintLibrary::ParseRawResponse(const FString& RawJson)
{
    FNPCDialogueContent Content;

    TSharedPtr<FJsonObject> JsonObject;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(RawJson);

    if (!FJsonSerializer::Deserialize(Reader, JsonObject) || !JsonObject.IsValid())
    {
        Content.Dialogue = RawJson;
        return Content;
    }

    Content.Dialogue = JsonObject->GetStringField(TEXT("dialogue"));
    Content.Emotion = JsonObject->GetStringField(TEXT("emotion"));
    Content.Action = JsonObject->GetStringField(TEXT("action"));

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
