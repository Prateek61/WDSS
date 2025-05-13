# Unreal Engine Plugin: GBuffer Extraction from Movie Render Queue

This plugin enables easy extraction of G-Buffers during Movie Render Queue (MRQ) rendering in **Unreal Engine 4.27**.  
It repackages and simplifies the workflow from [fuxihao66/ExtraNet](https://github.com/fuxihao66/ExtraNet/tree/main/EngineModificationGuide) into a reusable plugin—**no engine modification needed**.

Only tested on **UE 4.27**. Compatibility with other versions is not guaranteed.

---

## Setup & Usage

1. **Initialize the Plugin in Your Level**
   - Open your level blueprint.
   - On `BeginPlay`, call the `GBESetRenderTarget` node (under the **GBufferExtraction** category).
   - Pass in the included `RT_GBERenderTarget`.
   - Make sure to resize the `RT_GBERenderTarget` to match your desired render resolution.

2. **Use the Provided Movie Render Queue Preset**
   - Load the `GBEMoviePipelineConfig` preset (included with the plugin).
   - This preset is pre-configured to output G-Buffers (motion vector, normal, depth, roughness, etc.) correctly.

---

## Output Buffers
The plugin captures the following G-Buffers during rendering:

- Base Color
- World Normal
- Roughness, Metallic, Specular
- Scene Depth
- Motion Vectors
- Pre-tonemap HDR Color
- Normal
- NoV
