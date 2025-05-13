// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"
#include "MVProcessingViewExtension.h"
#include "Subsystems/EngineSubsystem.h"

class FGBufferExtractionModule : public IModuleInterface
{
public:

	/** IModuleInterface implementation */
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;

	void InitDelayed();

private:
	TSharedPtr<class FMVProcessingSceneViewExtension, ESPMode::ThreadSafe> MVProcessingViewExtension;
};
