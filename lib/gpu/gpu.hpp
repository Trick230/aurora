#pragma once

#include <aurora/aurora.h>
#include <aurora/math.hpp>

#include "rhi.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct SDL_Window;

namespace aurora::gpu {
struct GraphicsConfig {
  rhi::Format surfaceFormat;
  uint32_t surfaceWidth;
  uint32_t surfaceHeight;
  rhi::PresentMode presentMode;
  rhi::Format depthFormat;
  uint32_t msaaSamples;
  uint16_t textureAnisotropy;
};
struct Extent {
  uint32_t width = 0;
  uint32_t height = 0;
};
struct TextureWithSampler {
  std::shared_ptr<rhi::Texture> texture;
  std::shared_ptr<rhi::TextureView> view;
  Extent size;
  rhi::Format format = rhi::Format::Undefined;
  std::shared_ptr<rhi::Sampler> sampler;
};
struct Viewport {
  float left;
  float top;
  float width;
  float height;
  float znear;
  float zfar;

  bool operator==(const Viewport& rhs) const {
    return left == rhs.left && top == rhs.top && width == rhs.width && height == rhs.height && znear == rhs.znear &&
           zfar == rhs.zfar;
  }
  bool operator!=(const Viewport& rhs) const { return !(*this == rhs); }
};

extern rhi::Device* g_device;
extern rhi::Swapchain* g_surface;
extern GraphicsConfig g_graphicsConfig;
extern TextureWithSampler g_frameBuffer;
extern TextureWithSampler g_frameBufferResolved;
extern TextureWithSampler g_depthBuffer;
extern rhi::RenderPipeline* g_CopyPipeline;

bool initialize(AuroraBackend backend, bool allowCpu);
void shutdown();
void release_surface() noexcept;
bool refresh_surface(bool recreate = true);
void resize_swapchain(uint32_t width, uint32_t height, uint32_t nativeWidth, uint32_t nativeHeight, bool force = false);
TextureWithSampler create_render_texture(uint32_t width, uint32_t height, bool multisampled);
const TextureWithSampler& present_source() noexcept;
std::unique_ptr<rhi::BindGroup> create_copy_bind_group(const TextureWithSampler& source);
Viewport calculate_present_viewport(uint32_t surface_width, uint32_t surface_height, uint32_t content_width,
                                    uint32_t content_height) noexcept;
const TextureWithSampler& resample_present_source(rhi::CommandList& list, const Viewport& viewport);
// A pipeline's shaders from one HLSL source with vs_main and fs_main; fatal if it
// doesn't compile, since these sources are ours.
rhi::ShaderCode compile_shader(rhi::ShaderStage stage, const std::string& source, const char* entry);
// The HLSL sources of the present pipelines, for shipping them compiled.
std::vector<std::string> shader_sources();
} // namespace aurora::gpu
