#pragma once

#include "common.hpp"

namespace aurora::gfx::clear {
struct DrawData {
  PipelineRef pipeline;
  rhi::ClearColor color;
  float depth = 0.f;
  ClipRect rect{}; // in the target
};

constexpr uint32_t ClearPipelineConfigVersion = 2;
struct PipelineConfig {
  uint32_t version = ClearPipelineConfigVersion;
  uint32_t msaaSamples = 1;
  bool clearColor = true;
  bool clearAlpha = true;
  bool clearDepth = true;
  uint8_t _pad = 0;
};
static_assert(std::has_unique_object_representations_v<PipelineConfig>);

std::shared_ptr<rhi::RenderPipeline> create_pipeline(const PipelineConfig& config);
void render(const DrawData& data, rhi::CommandList& list, const Extent3D& targetSize);
} // namespace aurora::gfx::clear
