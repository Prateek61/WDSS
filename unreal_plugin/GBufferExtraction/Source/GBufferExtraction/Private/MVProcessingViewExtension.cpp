#include "MVProcessingViewExtension.h"

#include "GBEFunctionLibrary.h"
#include "MVProcessingShader.h"

#include "PostProcess/PostProcessing.h"
#include "PostProcess/PostProcessMaterial.h"
#include "Engine/TextureRenderTarget2D.h"
#include "SceneTextureParameters.h"

FMVProcessingSceneViewExtension::FMVProcessingSceneViewExtension(const FAutoRegister& AutoRegister)
	: FSceneViewExtensionBase(AutoRegister)
{
	UGBEFunctionLibrary::GBELogDebug("MVProcessingSceneViewExtension Constructor Called");
}

void FMVProcessingSceneViewExtension::PrePostProcessPass_RenderThread(FRDGBuilder& GraphBuilder, const FSceneView& View, const FPostProcessingInputs& Inputs)
{
	UTextureRenderTarget2D* RenderTarget = UGBEFunctionLibrary::GetRenderTarget();
	if (!RenderTarget)
	{
		return;
	}
	const FTextureRenderTargetResource* RenderTargetResource = RenderTarget->GetRenderTargetResource();
	if (!RenderTargetResource)
	{
		return;
	}
	const FTexture2DRHIRef& RenderTargetRHI = RenderTargetResource->GetRenderTargetTexture();
	if (!RenderTargetRHI)
	{
		return;
	}
	FRDGTextureRef RenderTargetTexture = GraphBuilder.RegisterExternalTexture(
		CreateRenderTarget(RenderTargetRHI, TEXT("GBE_MVProcessingRenderTarget")),
		TEXT("GBE_MVProcessingRenderTarget")
	);
	if (!RenderTargetTexture)
	{
		return;
	}

	// Depth and Velocity
	auto SceneTextures = Inputs.SceneTextures->GetParameters();
	FRDGTextureRef DepthTexture = SceneTextures->SceneDepthTexture;
	//FRDGTextureRef VelocityTexture = Inputs.GetInput(EPostProcessMaterialInput::Velocity).Texture;
	FRDGTextureRef VelocityTexture = SceneTextures->GBufferVelocityTexture;

	// Print the dimensions of velocity texture
	UGBEFunctionLibrary::GBELogDebug(FString::Printf(TEXT("Velocity Texture Dimensions: %d x %d"), VelocityTexture->Desc.Extent.X, VelocityTexture->Desc.Extent.Y));

	FRDGTextureRef DilatedVelocityTexture = AddMVProcessingPass(
		GraphBuilder,
		View,
		DepthTexture,
		VelocityTexture
	);

	// Add Copy Pass
	AddCopyTexturePass(
		GraphBuilder,
		DilatedVelocityTexture,
		RenderTargetTexture
	);

	return;
}
