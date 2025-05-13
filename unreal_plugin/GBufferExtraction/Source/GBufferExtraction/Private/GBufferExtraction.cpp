// Copyright Epic Games, Inc. All Rights Reserved.

#include "GBufferExtraction.h"
#include "GBEFunctionLibrary.h"


#define LOCTEXT_NAMESPACE "FGBufferExtractionModule"

void FGBufferExtractionModule::StartupModule()
{
	FString ShaderDir = FPaths::Combine(FPaths::ProjectPluginsDir(), TEXT("GBufferExtraction"), TEXT("Shaders"));
	AddShaderSourceDirectoryMapping(TEXT("/GBEShaders"), ShaderDir);

	UGBEFunctionLibrary::GBELogDebug("GBufferExtraction Module Started");

	FCoreDelegates::OnPostEngineInit.AddRaw(this, &FGBufferExtractionModule::InitDelayed);
}

void FGBufferExtractionModule::ShutdownModule()
{
	MVProcessingViewExtension.Reset();
	MVProcessingViewExtension = nullptr;
}

void FGBufferExtractionModule::InitDelayed()
{
	MVProcessingViewExtension = FSceneViewExtensions::NewExtension<FMVProcessingSceneViewExtension>();
}

#undef LOCTEXT_NAMESPACE
	
IMPLEMENT_MODULE(FGBufferExtractionModule, GBufferExtraction)