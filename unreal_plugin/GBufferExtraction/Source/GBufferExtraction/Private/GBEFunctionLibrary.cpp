#include "GBEFunctionLibrary.h"

#include "Engine/TextureRenderTarget2D.h"

static UTextureRenderTarget2D* GRenderTarget = nullptr;

UTextureRenderTarget2D* UGBEFunctionLibrary::GetRenderTarget()
{
	if (!GRenderTarget || !GRenderTarget->Resource)
	{
		return nullptr;
	}

	return GRenderTarget;
}

void UGBEFunctionLibrary::GBESetRenderTarget(UTextureRenderTarget2D* RenderTarget)
{
	if (RenderTarget)
	{
		GRenderTarget = RenderTarget;

		GBELogDebug("Render Target Set, Format: " + FString::Printf(TEXT("%s"), GetPixelFormatString(GRenderTarget->GetFormat())));
		GRenderTarget->UpdateResource();
	}
}

void UGBEFunctionLibrary::GBELogDebug(FString Message)
{
	UE_LOG(LogTemp, Log, TEXT("[GBufferExtraction] %s"), *Message);

	FString LogFilePath = FPaths::ProjectSavedDir() + TEXT("GBELog.txt");
	FFileHelper::SaveStringToFile(Message + LINE_TERMINATOR, *LogFilePath, FFileHelper::EEncodingOptions::AutoDetect, &IFileManager::Get(), FILEWRITE_Append);
}
