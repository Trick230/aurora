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

wgpu::RenderPipeline create_pipeline(const PipelineConfig& config) {
  const auto dstAlphaMode = dst_alpha_mode(config);
  dump_hlsl(config, dstAlphaMode);
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
