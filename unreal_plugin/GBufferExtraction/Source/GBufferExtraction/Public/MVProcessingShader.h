// From https://github.com/fuxihao66/ExtraNet/tree/main/EngineModificationGuide

#pragma once

#include "GlobalShader.h"
#include "SceneView.h"
#include "ShaderParameterStruct.h"
#include "ScreenPass.h"

class FMVProcessingCS : public FGlobalShader
{
public:
	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("USE_MV_PROCESSING"), 1);
	}

	DECLARE_GLOBAL_SHADER(FMVProcessingCS);
	SHADER_USE_PARAMETER_STRUCT(FMVProcessingCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		// Scene Texture Parameters
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D, SceneDepth)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D, SceneVelocity)

		SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
		SHADER_PARAMETER_STRUCT(FScreenPassTextureViewportParameters, InputInfo)

		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float4>, OutDilatedVelocity)
	END_SHADER_PARAMETER_STRUCT()
};

FRDGTextureRef AddMVProcessingPass(
	FRDGBuilder& GraphBuilder,
	const FSceneView& View,
	FRDGTextureRef SceneDepth,
	FRDGTextureRef SceneVelocity
);
