#pragma once

#include "../gfx/common.hpp"
#include "gx.hpp"

namespace aurora::gx {
constexpr uint32_t GXPipelineConfigVersion = 18;
struct PipelineConfig {
  uint32_t version = GXPipelineConfigVersion;
  uint32_t msaaSamples = 1;
  ShaderConfig shaderConfig;
  GXCompare depthFunc;
  GXCullMode cullMode;
  GXBlendMode blendMode;
  GXBlendFactor blendFacSrc, blendFacDst;
  GXLogicOp blendOp;
  uint32_t dstAlpha;
  bool depthCompare, depthUpdate, alphaUpdate, colorUpdate;
};
static_assert(std::has_unique_object_representations_v<PipelineConfig>);

std::shared_ptr<rhi::RenderPipeline> create_pipeline(const PipelineConfig& config);
DstAlphaMode dst_alpha_mode(const PipelineConfig& config) noexcept;

// The ubershader's pipeline for a draw's: the fixed-function state alone.
PipelineConfig uber_pipeline_config(const PipelineConfig& config) noexcept;
std::shared_ptr<rhi::RenderPipeline> create_uber_pipeline(const PipelineConfig& config);
// The draw's configuration in the frame's storage, for its uniform block: its
// offset in words.
u32 push_uber_config(const ShaderConfig& config, const ShaderInfo& info) noexcept;
void render(const DrawData& data, rhi::CommandList& list);

void queue_surface(const u8* dlStart, uint32_t dlSize, bool bigEndian) noexcept;
} // namespace aurora::gx
