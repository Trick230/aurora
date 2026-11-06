#pragma once

#include "common.hpp"

#include <dolphin/gx/GXEnum.h>

namespace aurora::gfx::tex_copy_conv {

enum class SampleFilter : uint8_t {
  Nearest,
  Linear,
};

struct ConvRequest {
  GXTexFmt fmt;
  std::shared_ptr<rhi::TextureView> srcView; // View of resolved EFB / offscreen color/depth
  Range uniformRange;        // UV transform uniform (offset + scale)
  TextureHandle dst;         // Destination texture
  SampleFilter sampleFilter = SampleFilter::Nearest;
  bool multisampled = false; // depth only: srcView is the multisampled EFB depth (sample 0 is read)
};

bool needs_conversion(GXTexFmt fmt);

void initialize();
void shutdown();
void run(rhi::CommandList& list, const ConvRequest& req);
void blit(rhi::CommandList& list, const ConvRequest& req);
// The HLSL sources of its pipelines, for shipping them compiled.
std::vector<std::string> shader_sources();

} // namespace aurora::gfx::tex_copy_conv
