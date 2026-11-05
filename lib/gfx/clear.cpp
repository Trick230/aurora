#include "clear.hpp"

#include "../gpu/gpu.hpp"

#include <mutex>

namespace aurora::gfx::clear {

using gpu::g_device;
using gpu::g_graphicsConfig;

static constexpr const char* ClearShader = R"(
static const float2 positions[3] = {float2(-1.0, 1.0), float2(-1.0, -3.0), float2(3.0, 1.0)};
float4 vs_main(uint vtxIdx : SV_VertexID) : SV_Position {
    return float4(positions[vtxIdx], 0.0, 1.0);
}
float4 fs_main() : SV_Target {
    return float4(1.0, 1.0, 1.0, 1.0);
}
)";

std::shared_ptr<rhi::RenderPipeline> create_pipeline(const PipelineConfig& config) {
  static std::mutex mutex;
  static std::unique_ptr<rhi::PipelineLayout> layout;
  static rhi::ShaderCode vertex, fragment;
  {
    std::scoped_lock lock{mutex};
    if (!layout) {
      layout = g_device->create_pipeline_layout({nullptr, 0});
      vertex = gpu::compile_shader(rhi::ShaderStage::Vertex, ClearShader, "vs_main");
      fragment = gpu::compile_shader(rhi::ShaderStage::Fragment, ClearShader, "fs_main");
    }
  }
  rhi::RenderPipelineDesc desc;
  desc.layout = layout.get();
  desc.vertex = &vertex;
  desc.vertex_entry = "vs_main";
  desc.fragment = &fragment;
  desc.fragment_entry = "fs_main";
  desc.color_format = g_graphicsConfig.surfaceFormat;
  desc.color_write = (config.clearColor ? rhi::WriteR | rhi::WriteG | rhi::WriteB : 0) | (config.clearAlpha ? rhi::WriteA : 0);
  desc.blend = true;
  desc.blend_color = {rhi::BlendFactor::Constant, rhi::BlendFactor::Zero, rhi::BlendOp::Add};
  desc.blend_alpha = desc.blend_color;
  desc.depth_format = g_graphicsConfig.depthFormat;
  desc.depth_write = config.clearDepth;
  desc.depth_compare = rhi::Compare::Always;
  desc.samples = config.msaaSamples;
  return g_device->create_render_pipeline(desc);
}

void render(const DrawData& data, rhi::CommandList& list, const Extent3D& targetSize) {
  if (!bind_pipeline(data.pipeline, list)) {
    return;
  }
  list.set_blend_constant(data.color.r, data.color.g, data.color.b, data.color.a);
  list.set_viewport(0.f, 0.f, static_cast<float>(targetSize.width), static_cast<float>(targetSize.height), data.depth,
                    data.depth);
  list.set_scissor(data.rect.x, data.rect.y, data.rect.width, data.rect.height);
  list.draw(3, 0);
}
} // namespace aurora::gfx::clear
