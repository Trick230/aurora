#pragma once

#include "common.hpp"

#include <string>
#include <vector>

namespace aurora::gfx::ao {
// Ground-truth ambient occlusion (GTAO) from the EFB's depth, darkening the
// world's colors once it has been drawn and before the 2D on top of it.

void initialize();
void shutdown();
bool enabled() noexcept;

// The GX fog the world was drawn with: its type (0: none) and a, b, c.
struct Fog {
  uint32_t type = 0;
  float a = 0.f, b = 0.f, c = 0.f;
};
// The uniform block for one application: the projection, viewport and fog the
// world was drawn with, and the EFB's size.
Range push_params(const Mat4x4<float>& proj, const Viewport& viewport, const Extent3D& target, const Fog& fog);

struct Target {
  rhi::Texture* color = nullptr;
  rhi::TextureView* colorView = nullptr; // read before the occlusion goes over it
  rhi::Texture* resolve = nullptr; // with MSAA, the color's resolve target
  rhi::TextureView* depth = nullptr;
  uint32_t samples = 1;      // the color's
  uint32_t depthSamples = 1;
  Extent3D size;
  ClipRect rect; // the world's viewport
  Range params;
};
// After the pass that drew the world, outside any render pass.
void apply(rhi::CommandList& list, const Target& target);

// The HLSL sources of its pipelines, for shipping them compiled.
std::vector<std::string> shader_sources();
} // namespace aurora::gfx::ao
