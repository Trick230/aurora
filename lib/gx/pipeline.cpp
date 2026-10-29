#include "pipeline.hpp"

#include "../gfx/frame_interp.hpp"
#include "../webgpu/gpu.hpp"
#include "gx_fmt.hpp"
#include "shader_info.hpp"

namespace aurora::gx {
static Module Log("aurora::gx");

wgpu::RenderPipeline create_pipeline(const PipelineConfig& config) {
  // A destination alpha other than 0 (which the blend gives with any shader
  // alpha) needs the shader's alpha to be 1; a color blend that reads source
  // alpha then reads the TEV's from a second output.
  auto dstAlphaMode = DstAlphaMode::None;
  if (config.alphaUpdate && config.dstAlpha != UINT32_MAX && config.dstAlpha != 0) {
    const auto readsSourceAlpha = [](GXBlendFactor factor) {
      return factor == GX_BL_SRCALPHA || factor == GX_BL_INVSRCALPHA;
    };
    const bool needsSourceAlpha = config.colorUpdate && config.blendMode == GX_BM_BLEND &&
                                  (readsSourceAlpha(config.blendFacSrc) || readsSourceAlpha(config.blendFacDst));
    dstAlphaMode = needsSourceAlpha && webgpu::g_dualSourceBlendingSupported ? DstAlphaMode::DualSource
                                                                             : DstAlphaMode::Replace;
    if (needsSourceAlpha && !webgpu::g_dualSourceBlendingSupported) {
      static bool warned = false;
      if (!warned) {
        warned = true;
        Log.warn("no dual source blending: blends under GXSetDstAlpha read source alpha 1");
      }
    }
  }
  const auto shader = build_shader(config.shaderConfig, dstAlphaMode);
  const auto label = fmt::format("GX Pipeline {:x} shader {:x}",
                                 xxh3_hash(config, static_cast<HashType>(gfx::ShaderType::GX)),
                                 xxh3_hash(config.shaderConfig));
  return build_pipeline(config, dstAlphaMode, {}, shader, label.c_str());
}

template <typename Encoder>
static void encode_draw(const DrawData& data, const Encoder& pass) {
  if (data.cullMode == GX_CULL_ALL) {
    return;
  }
  if (!gfx::bind_pipeline(data.pipeline, pass)) {
    return;
  }

  const std::array offsets{data.uniformRange.offset};
  pass.SetBindGroup(1, *gfx::interp::g_vertexUniformBindGroup, offsets.size(), offsets.data());
  if (data.bindGroups.textureBindGroup) {
    gfx::bind_texture_group(data.bindGroups.textureBindGroup, pass);
  }
  if constexpr (std::is_same_v<Encoder, wgpu::RenderPassEncoder>) {
    if (data.dstAlpha != UINT32_MAX) {
      const wgpu::Color color{0.f, 0.f, 0.f, data.dstAlpha / 255.f};
      pass.SetBlendConstant(&color);
    }
  }
  if (data.indexCount == 0) {
    pass.Draw(data.vtxCount, data.instanceCount, 0, data.firstInstance);
  } else {
    gfx::bind_index_buffer(pass);
    pass.DrawIndexed(data.indexCount, data.instanceCount, data.idxRange.offset / sizeof(uint16_t), 0,
                     data.firstInstance);
  }
}

void render(const DrawData& data, const wgpu::RenderPassEncoder& pass) { encode_draw(data, pass); }

void render(const DrawData& data, const wgpu::RenderBundleEncoder& bundle) { encode_draw(data, bundle); }
} // namespace aurora::gx
