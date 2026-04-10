// Copyright NPC Engine. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

/**
 * NPC Engine plugin module interface.
 */
class FNPCEngineModule : public IModuleInterface
{
public:
    /** Called when the module is loaded into memory. */
    virtual void StartupModule() override;

    /** Called when the module is unloaded from memory. */
    virtual void ShutdownModule() override;
};
