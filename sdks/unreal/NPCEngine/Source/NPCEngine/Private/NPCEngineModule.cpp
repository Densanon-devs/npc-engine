// Copyright NPC Engine. All Rights Reserved.

#include "NPCEngineModule.h"
#include "Modules/ModuleManager.h"

#define LOCTEXT_NAMESPACE "FNPCEngineModule"

void FNPCEngineModule::StartupModule()
{
    // Module startup — nothing special needed.
    UE_LOG(LogTemp, Log, TEXT("NPC Engine plugin loaded."));
}

void FNPCEngineModule::ShutdownModule()
{
    // Module shutdown — nothing special needed.
    UE_LOG(LogTemp, Log, TEXT("NPC Engine plugin unloaded."));
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FNPCEngineModule, NPCEngine)
