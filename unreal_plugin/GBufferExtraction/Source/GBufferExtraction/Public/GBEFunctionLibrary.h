#pragma once

#include "Kismet/BlueprintFunctionLibrary.h"
#include "MVProcessingViewExtension.h"
#include "GBEFunctionLibrary.generated.h"

UCLASS()
class GBUFFEREXTRACTION_API UGBEFunctionLibrary : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
	static UTextureRenderTarget2D* GetRenderTarget();

	UFUNCTION(BlueprintCallable, Category = "GBufferExtraction")
	static void GBESetRenderTarget(UTextureRenderTarget2D* RenderTarget);

	UFUNCTION(BlueprintCallable, Category = "GBufferExtraction")
	static void GBELogDebug(FString Message);
};