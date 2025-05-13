#include "MVProcessingViewExtension.h"

#include "GBEFunctionLibrary.h"
#include "MVProcessingShader.h"

#include "PostProcess/PostProcessing.h"
#include "PostProcess/PostProcessMaterial.h"
#include "PostProcess/PostProcessMaterial.h"
#include "Engine/TextureRenderTarget2D.h"
#include "SceneTextureParameters.h"

FMVProcessingSceneViewExtension::FMVProcessingSceneViewExtension(const FAutoRegister& AutoRegister)
	: FSceneViewExtensionBase(AutoRegister)
{
	UGBEFunctionLibrary::GBELogDebug("MVProcessingSceneViewExtension Constructor Called");
}

void FMVProcessingSceneViewExtension::SubscribeToPostProcessingPass(EPostProcessingPass Pass, FAfterPassCallbackDelegateArray& InOutPassCallbacks, bool bIsPassEnabled)
{
	if (Pass == EPostProcessingPass::Tonemap)
	{
		InOutPassCallbacks.Add(
			FAfterPassCallbackDelegate::CreateRaw(
				this,
				&FMVProcessingSceneViewExtension::AfterTonemapPass
			)
		);
	}
}

FScreenPassTexture FMVProcessingSceneViewExtension::AfterTonemapPass(FRDGBuilder& GraphBuilder, const FSceneView& View, const FPostProcessMaterialInputs& Inputs)
{
	UTextureRenderTarget2D* RenderTarget = UGBEFunctionLibrary::GetRenderTarget();
	if (!RenderTarget)
	{
		return Inputs.GetInput(EPostProcessMaterialInput::SceneColor);
	}
	const FTextureRenderTargetResource* RenderTargetResource = RenderTarget->GetRenderTargetResource();
	if (!RenderTargetResource)
	{
		return Inputs.GetInput(EPostProcessMaterialInput::SceneColor);
	}
	const FTexture2DRHIRef& RenderTargetRHI = RenderTargetResource->GetRenderTargetTexture();
	if (!RenderTargetRHI)
	{
		return Inputs.GetInput(EPostProcessMaterialInput::SceneColor);
	}
	FRDGTextureRef RenderTargetTexture = GraphBuilder.RegisterExternalTexture(
		CreateRenderTarget(RenderTargetRHI, TEXT("GBE_MVProcessingRenderTarget")),
		TEXT("GBE_MVProcessingRenderTarget")
	);
	if (!RenderTargetTexture)
	{
		return Inputs.GetInput(EPostProcessMaterialInput::SceneColor);
	}

	// Depth and Velocity
	FRDGTextureRef DepthTexture = Inputs.SceneTextures.SceneTextures->GetParameters()->SceneDepthTexture;
	FRDGTextureRef VelocityTexture = Inputs.GetInput(EPostProcessMaterialInput::Velocity).Texture;

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

	return Inputs.GetInput(EPostProcessMaterialInput::SceneColor);
}
