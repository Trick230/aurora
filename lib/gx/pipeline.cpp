#include "pipeline.hpp"

#include "../gfx/frame_interp.hpp"
#include "../webgpu/gpu.hpp"
#include "gx_fmt.hpp"
#include "shader_info.hpp"
#include "tracy/Tracy.hpp"

namespace aurora::gx {
static Module Log("aurora::gx");

wgpu::RenderPipeline create_pipeline(const PipelineConfig& config) {
  ZoneScoped;
  const auto shader = build_shader(config.shaderConfig);
  const auto label = fmt::format("GX Pipeline {:x} shader {:x}",
                                 xxh3_hash(config, static_cast<HashType>(gfx::ShaderType::GX)),
                                 xxh3_hash(config.shaderConfig));
  return build_pipeline(config, {}, shader, label.c_str());
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
    pass.SetBindGroup(2, gfx::find_bind_group(data.bindGroups.textureBindGroup));
  }
  pass.SetIndexBuffer(gfx::g_indexBuffer, wgpu::IndexFormat::Uint16, data.idxRange.offset, data.idxRange.size);
  if constexpr (std::is_same_v<Encoder, wgpu::RenderPassEncoder>) {
    if (data.dstAlpha != UINT32_MAX) {
      const wgpu::Color color{0.f, 0.f, 0.f, data.dstAlpha / 255.f};
      pass.SetBlendConstant(&color);
    }
  }
  if (data.indexCount == 0) {
    pass.Draw(data.vtxCount, data.instanceCount, 0, data.firstInstance);
  } else {
    pass.DrawIndexed(data.indexCount, data.instanceCount, 0, 0, data.firstInstance);
  }
}

void render(const DrawData& data, const wgpu::RenderPassEncoder& pass) { encode_draw(data, pass); }

void render(const DrawData& data, const wgpu::RenderBundleEncoder& bundle) { encode_draw(data, bundle); }
} // namespace aurora::gx
