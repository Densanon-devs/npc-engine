# NPC Engine - Unreal Engine Plugin

HTTP client wrapper for the NPC Engine REST API. Provides Blueprint and C++ access to AI-powered NPC dialogue with trust, gossip, emotions, and quests.

## Installation

1. Copy the `NPCEngine/` folder into your project's `Plugins/` directory.
2. Add `"NPCEngine"` to your `.uproject` file's `Plugins` array:

```json
{
    "Plugins": [
        {
            "Name": "NPCEngine",
            "Enabled": true
        }
    ]
}
```

3. If using C++, add `"NPCEngine"` to your module's `Build.cs` dependencies:

```csharp
PublicDependencyModuleNames.Add("NPCEngine");
```

4. Regenerate project files and build.

## Requirements

- Unreal Engine 5.x
- NPC Engine server running on `localhost:8000` (configurable)

## Blueprint Usage

1. Create a `UNPCEngineClient` variable (e.g., in your PlayerController).
2. In `BeginPlay`, construct one with `ConstructObject` or `NewObject`.
3. Bind delegates to handle responses (`OnGenerateResponse`, `OnNPCList`, etc.).
4. Call methods like `Generate`, `ListNPCs`, `AdjustTrust`, etc.

Example (Blueprint pseudocode):
```
BeginPlay:
  NPCClient = Construct Object (class: NPCEngineClient)
  Bind Event to OnGenerateResponse → HandleDialogue
  Bind Event to OnRequestFailed → HandleError

On Player Input:
  NPCClient → Generate(Prompt: "Hello there!", NpcId: "blacksmith")

HandleDialogue(Response):
  DialogueText = Get Dialogue(Response)
  EmotionText = Get Emotion(Response)
  // Display in UI widget
```

## C++ Usage

```cpp
#include "NPCEngineClient.h"

// In your actor or component:
void AMyActor::BeginPlay()
{
    Super::BeginPlay();

    NPCClient = NewObject<UNPCEngineClient>(this);
    NPCClient->ServerUrl = TEXT("http://127.0.0.1:8000");

    NPCClient->OnGenerateResponse.AddDynamic(this, &AMyActor::OnDialogueReceived);
    NPCClient->OnRequestFailed.AddDynamic(this, &AMyActor::OnNPCError);
}

void AMyActor::TalkToNPC(const FString& Prompt)
{
    NPCClient->Generate(Prompt, TEXT("blacksmith"));
}

void AMyActor::OnDialogueReceived(const FNPCGenerateResponse& Response)
{
    UE_LOG(LogTemp, Log, TEXT("NPC says: %s"), *Response.Parsed.Dialogue);
    UE_LOG(LogTemp, Log, TEXT("Emotion: %s"), *Response.Parsed.Emotion);
}

void AMyActor::OnNPCError(const FString& Endpoint, const FString& Error)
{
    UE_LOG(LogTemp, Error, TEXT("NPC Engine error on %s: %s"), *Endpoint, *Error);
}
```

## API Endpoints

| Method | Description |
|--------|-------------|
| `Generate(Prompt, NpcId)` | Generate NPC dialogue |
| `ListNPCs()` | List all available NPCs |
| `SwitchNPC(NpcId)` | Switch active NPC |
| `InjectEvent(Description, NpcId)` | Inject world event |
| `AdjustTrust(NpcId, Delta, Reason)` | Adjust NPC trust |
| `SetMood(NpcId, Mood, Intensity)` | Set NPC mood |
| `AddScratchpad(NpcId, Text, Importance)` | Add scratchpad entry |
| `AcceptQuest(QuestId, QuestName, GivenBy)` | Accept a quest |
| `CompleteQuest(QuestId)` | Complete a quest |
| `CheckHealth()` | Server health check |

## Configuration

Set `ServerUrl` on the `UNPCEngineClient` instance to point to your NPC Engine server. Defaults to `http://127.0.0.1:8000`.
