#include "shader_hlsl.hpp"

#include <cstdio>
#include <cstdlib>

#include "pipeline.hpp"

#include "../gfx/frame_interp.hpp"
#include "../webgpu/gpu.hpp"
#include "gx_fmt.hpp"
#include "shader_info.hpp"

namespace aurora::gx {
static Module Log("aurora::gx");

DstAlphaMode dst_alpha_mode(const PipelineConfig& config) noexcept {
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
    dstAlphaMode = needsSourceAlpha ? DstAlphaMode::DualSource : DstAlphaMode::Replace;
  }
  return dstAlphaMode;
}

static void dump_uber_hlsl(const char* dir) noexcept;

// With AURORA_HLSL_DUMP=<dir>, the HLSL of every pipeline the game asks for, for
// tests/rhi's check_hlsl: the corpus until the game itself renders through it.
static void dump_hlsl(const PipelineConfig& config, DstAlphaMode dstAlphaMode) noexcept {
  static const char* const dir = std::getenv("AURORA_HLSL_DUMP");
  if (dir == nullptr) {
    return;
  }
  static const bool uberDumped = (dump_uber_hlsl(dir), true);
  (void)uberDumped;
  const auto source = hlsl::build_shader_source(config.shaderConfig, dstAlphaMode);
  const auto path = fmt::format("{}/{:016x}_{}.hlsl", dir, xxh3_hash(config.shaderConfig), static_cast<int>(dstAlphaMode));
  if (std::FILE* file = std::fopen(path.c_str(), "wb")) {
    std::fwrite(source.data(), 1, source.size(), file);
    std::fclose(file);
  }
}

// The ubershader's three sources, the first time the dump runs.
static void dump_uber_hlsl(const char* dir) noexcept {
  for (const auto mode : {DstAlphaMode::None, DstAlphaMode::Replace, DstAlphaMode::DualSource}) {
    const auto source = hlsl::uber_source(mode);
    const auto path = fmt::format("{}/uber_{}.hlsl", dir, static_cast<int>(mode));
    if (std::FILE* file = std::fopen(path.c_str(), "wb")) {
      std::fwrite(source.data(), 1, source.size(), file);
      std::fclose(file);
    }
  }
}

std::shared_ptr<rhi::RenderPipeline> create_pipeline(const PipelineConfig& config) {
  const auto dstAlphaMode = dst_alpha_mode(config);
  dump_hlsl(config, dstAlphaMode);
  const auto source = hlsl::build_shader_source(config.shaderConfig, dstAlphaMode);
  const auto vertex = webgpu::compile_shader(rhi::ShaderStage::Vertex, source, "vs_main");
  const auto fragment = webgpu::compile_shader(rhi::ShaderStage::Fragment, source, "fs_main");
  return build_pipeline(config, dstAlphaMode, vertex, fragment);
}

void render(const DrawData& data, rhi::CommandList& list) {
  if (data.cullMode == GX_CULL_ALL) {
    return;
  }
  if (!gfx::bind_pipeline(data.pipeline, list)) {
    return;
  }
  gfx::bind_gx_groups(list);
  const uint32_t offset = data.uniformRange.offset;
  list.set_bind_group(1, *gfx::interp::g_vertexUniformBindGroup, &offset, 1);
  if (data.bindGroups.textureBindGroup) {
    gfx::bind_texture_group(data.bindGroups.textureBindGroup, list);
  }
  if (data.dstAlpha != UINT32_MAX) {
    list.set_blend_constant(0.f, 0.f, 0.f, data.dstAlpha / 255.f);
  }
  list.set_draw_constant(data.firstInstance);
  if (data.indexCount == 0) {
    list.draw(data.vtxCount, 0, data.instanceCount);
  } else {
    gfx::bind_index_buffer(list);
    list.draw_indexed(data.indexCount, data.idxRange.offset / sizeof(uint16_t), 0, data.instanceCount);
  }
}
} // namespace aurora::gx
