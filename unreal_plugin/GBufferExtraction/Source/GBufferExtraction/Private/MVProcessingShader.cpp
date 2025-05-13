// From https://github.com/fuxihao66/ExtraNet/tree/main/EngineModificationGuide

#include "MVProcessingShader.h"

#include "RenderGraphUtils.h"

IMPLEMENT_GLOBAL_SHADER(
	FMVProcessingCS,
	"/GBEShaders/MotionVectorProcessing.usf",
	"MainCS",
	SF_Compute
);

FRDGTextureRef AddMVProcessingPass(
	FRDGBuilder& GraphBuilder,
	const FSceneView& View,
	FRDGTextureRef SceneDepth,
	FRDGTextureRef SceneVelocity
)
{
	FRDGTextureDesc FloatDesc = FRDGTextureDesc::Create2DDesc(
		SceneDepth->Desc.Extent,
		PF_FloatRGBA,
		FClearValueBinding::Black,
		/* InFlags = */ TexCreate_None,
		/* InTargetableFlags = */ TexCreate_ShaderResource | TexCreate_UAV,
		/* bInForceSeparateTargetAndShaderResource = */ false
	);
	FRDGTextureRef DilatedVelocityTexture = GraphBuilder.CreateTexture(FloatDesc, TEXT("DilatedVelocityTexture"), ERDGTextureFlags::MultiFrame);

	FMVProcessingCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FMVProcessingCS::FParameters>();

	PassParameters->SceneDepth = SceneDepth;
	PassParameters->SceneVelocity = SceneVelocity;

	PassParameters->View = View.ViewUniformBuffer;
	const FScreenPassTextureViewport Viewport(FloatDesc.Extent);
	PassParameters->InputInfo = GetScreenPassTextureViewportParameters(Viewport);

	PassParameters->OutDilatedVelocity = GraphBuilder.CreateUAV(DilatedVelocityTexture);

	TShaderMapRef<FMVProcessingCS> ComputeShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));

	FComputeShaderUtils::AddPass(
		GraphBuilder,
		RDG_EVENT_NAME("MotionVectorCS"),
		ComputeShader,
		PassParameters,
		FComputeShaderUtils::GetGroupCount(FloatDesc.Extent, 8)
	);

	return DilatedVelocityTexture;
}