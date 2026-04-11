# Anima — Unreal Engine Plugin

*Every NPC has a soul.*

Blueprint and C++ client for the Anima NPC system. AI-powered dialogue with trust, gossip, emotions, and quests.

## Setup

### 1. Download Anima server

Download the Anima server binary for your platform from [GitHub Releases](https://github.com/Densanon-devs/npc-engine/releases). Place it in your project:

```
YourProject/
  Binaries/
    NPCEngine/
      npc-engine.exe          (Windows)
      models/
        qwen2.5-0.5b-instruct-q4_k_m.gguf    (from HuggingFace)
      data/
        worlds/                (your NPC profiles)
      config.yaml
```

Download the AI model (~469MB) from [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF).

### 2. Install the plugin

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

### 3. Launch Anima with your game

In your GameMode's `BeginPlay`, launch the server binary as a subprocess. See the [full documentation](https://github.com/Densanon-devs/npc-engine) for details.

When shipping, include the `Binaries/NPCEngine/` folder in your build. The binary runs as an invisible background process — players never see it.

## Requirements

- Unreal Engine 5.x
- Anima server binary (from GitHub Releases) running on `localhost:8000`

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
