#include "gpu.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <string>

#include <SDL3/SDL.h>
#include <aurora/aurora.h>
#include <aurora/gfx.h>

#include "../gfx/common.hpp"
#include "../gfx/frame_interp.hpp"
#include "../gfx/render_worker.hpp"
#include "../internal.hpp"
#include "../window.hpp"
#include "shader_cache.hpp"

namespace aurora::gpu {
static Module Log("aurora::gpu");

static std::unique_ptr<rhi::Device> s_device;
static std::unique_ptr<rhi::Swapchain> s_surface;
rhi::Device* g_device = nullptr;
rhi::Swapchain* g_surface = nullptr;
GraphicsConfig g_graphicsConfig;
TextureWithSampler g_frameBuffer;
TextureWithSampler g_frameBufferResolved;
TextureWithSampler g_depthBuffer;

// EFB -> XFB copy pipeline
static std::unique_ptr<rhi::BindGroupLayout> g_CopyBindGroupLayout;
static std::unique_ptr<rhi::PipelineLayout> g_CopyPipelineLayout;
static std::unique_ptr<rhi::RenderPipeline> s_CopyPipeline;
rhi::RenderPipeline* g_CopyPipeline = nullptr;
static std::unique_ptr<rhi::BindGroupLayout> g_ResampleBindGroupLayout;
static std::unique_ptr<rhi::PipelineLayout> g_ResamplePipelineLayout;
static std::unique_ptr<rhi::RenderPipeline> g_ResamplePipeline;
static std::unique_ptr<rhi::Buffer> g_ResampleUniformBuffer;
static TextureWithSampler g_resampledFrameBuffer;

namespace {
struct ResampleUniformBlock {
  uint32_t samplerMode = 0;
  float frameWidth = 0.f;
  float frameHeight = 0.f;
};
// More than frames can be in flight: each present takes the next slot.
constexpr uint32_t ResampleUniformStride = 256;
constexpr uint32_t ResampleUniformSlots = 8;

constexpr const char* FullscreenVertex = R"(
struct VertexOutput {
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};
static const float2 positions[3] = {float2(-1.0, 1.0), float2(-1.0, -3.0), float2(3.0, 1.0)};
static const float2 uvs[3] = {float2(0.0, 0.0), float2(0.0, 2.0), float2(2.0, 0.0)};
VertexOutput vs_main(uint vtxIdx : SV_VertexID) {
    VertexOutput output;
    output.pos = float4(positions[vtxIdx], 0.0, 1.0);
    output.uv = uvs[vtxIdx];
    return output;
}
)";

constexpr const char* CopyShader = R"(
SamplerState efb_sampler : register(s0, space0);
Texture2D efb_texture : register(t1, space0);
float4 fs_main(VertexOutput input) : SV_Target {
    return float4(efb_texture.Sample(efb_sampler, input.uv).rgb, 1.0);
}
)";

constexpr const char* ResampleShader = R"(
struct Uniforms {
    uint sampler_mode;
    float frame_width;
    float frame_height;
};
ConstantBuffer<Uniforms> uniforms : register(b0, space0);
SamplerState s : register(s1, space0);
Texture2D t : register(t2, space0);

float4 sample_by_pixel(int2 pixel) {
    uint width, height;
    t.GetDimensions(width, height);
    int2 coord = clamp(pixel, int2(0, 0), int2(width, height) - int2(1, 1));
    return t.Load(int3(coord, 0));
}

float4 sample_area(float4 frag_position) {
    uint width, height;
    t.GetDimensions(width, height);
    float2 source_size = float2(width, height);
    float2 target_size = max(float2(uniforms.frame_width, uniforms.frame_height), float2(1.0, 1.0));

    float2 source_min = clamp((frag_position.xy - float2(0.5, 0.5)) / target_size, float2(0.0, 0.0), float2(1.0, 1.0)) * source_size;
    float2 source_max = clamp((frag_position.xy + float2(0.5, 0.5)) / target_size, float2(0.0, 0.0), float2(1.0, 1.0)) * source_size;

    int2 first_pixel = int2(floor(source_min));
    int2 last_pixel = int2(ceil(source_max));
    // Only the pixels the footprint covers (at most 16 a side).
    int2 end_pixel = min(last_pixel, first_pixel + int2(16, 16));

    float4 avg_color = float4(0.0, 0.0, 0.0, 0.0);
    float total_weight = 0.0;
    for (int source_y = first_pixel.y; source_y < end_pixel.y; source_y++) {
        float y0 = float(source_y);
        float weight_y = max(min(source_max.y, y0 + 1.0) - max(source_min.y, y0), 0.0);
        for (int source_x = first_pixel.x; source_x < end_pixel.x; source_x++) {
            float x0 = float(source_x);
            float weight_x = max(min(source_max.x, x0 + 1.0) - max(source_min.x, x0), 0.0);
            float weight = weight_x * weight_y;
            avg_color += weight * sample_by_pixel(int2(source_x, source_y));
            total_weight += weight;
        }
    }
    return avg_color / max(total_weight, 0.000001);
}

float4 fs_main(VertexOutput input) : SV_Target {
    float4 color = t.Sample(s, input.uv);
    if (uniforms.sampler_mode == 1u) {
        color = sample_area(input.pos);
    }
    return float4(color.rgb, 1.0);
}
)";

uint32_t sampler_mode(AuroraSampler sampler) noexcept {
  switch (sampler) {
  case SAMPLER_AREA:
    return 1;
  case SAMPLER_BILINEAR:
  default:
    return 0;
  }
}

uint32_t viewport_extent(float value) noexcept {
  return std::max(1u, static_cast<uint32_t>(std::lround(std::max(value, 1.f))));
}

rhi::PresentMode best_present_mode(bool vsync) { return vsync ? rhi::PresentMode::Fifo : rhi::PresentMode::Mailbox; }

std::shared_ptr<rhi::Sampler> create_clamped_sampler() {
  rhi::SamplerDesc desc;
  desc.u = rhi::AddressMode::ClampToEdge;
  desc.v = rhi::AddressMode::ClampToEdge;
  return g_device->create_sampler(desc);
}

std::unique_ptr<rhi::RenderPipeline> create_present_pipeline(rhi::PipelineLayout* layout, const char* shader) {
  const std::string source = std::string{FullscreenVertex} + shader;
  const auto vertex = compile_shader(rhi::ShaderStage::Vertex, source, "vs_main");
  const auto fragment = compile_shader(rhi::ShaderStage::Fragment, source, "fs_main");
  rhi::RenderPipelineDesc desc;
  desc.layout = layout;
  desc.vertex = &vertex;
  desc.vertex_entry = "vs_main";
  desc.fragment = &fragment;
  desc.fragment_entry = "fs_main";
  desc.color_format = g_graphicsConfig.surfaceFormat;
  return g_device->create_render_pipeline(desc);
}

void create_copy_pipeline() {
  const std::array entries{
      rhi::BindingLayout{0, rhi::BindingKind::Sampler},
      rhi::BindingLayout{1, rhi::BindingKind::Texture},
  };
  g_CopyBindGroupLayout = g_device->create_bind_group_layout({entries.data(), entries.size()});
  rhi::BindGroupLayout* groups[] = {g_CopyBindGroupLayout.get()};
  g_CopyPipelineLayout = g_device->create_pipeline_layout({groups, 1});
  s_CopyPipeline = create_present_pipeline(g_CopyPipelineLayout.get(), CopyShader);
  g_CopyPipeline = s_CopyPipeline.get();
}

void create_resample_pipeline() {
  rhi::BindingLayout uniform{0, rhi::BindingKind::Uniform};
  uniform.dynamic_offset = true;
  const std::array entries{
      uniform,
      rhi::BindingLayout{1, rhi::BindingKind::Sampler},
      rhi::BindingLayout{2, rhi::BindingKind::Texture},
  };
  g_ResampleBindGroupLayout = g_device->create_bind_group_layout({entries.data(), entries.size()});
  rhi::BindGroupLayout* groups[] = {g_ResampleBindGroupLayout.get()};
  g_ResamplePipelineLayout = g_device->create_pipeline_layout({groups, 1});
  g_ResamplePipeline = create_present_pipeline(g_ResamplePipelineLayout.get(), ResampleShader);
  g_ResampleUniformBuffer =
      g_device->create_buffer({ResampleUniformStride * ResampleUniformSlots, rhi::MemoryKind::Upload});
}
} // namespace

rhi::ShaderCode compile_shader(rhi::ShaderStage stage, const std::string& source, const char* entry) {
  rhi::ShaderCode code;
  if (shader_cache::find(stage, source, entry, &code)) {
    return code;
  }
  std::string error;
  if (!rhi::compile_shader(g_device->backend(), stage, source, entry, &code, &error)) {
    FATAL("Shader {} doesn't compile: {}\n{}", entry, error, source);
  }
  shader_cache::store(stage, source, entry, code);
  return code;
}

TextureWithSampler create_render_texture(uint32_t width, uint32_t height, bool multisampled) {
  if (width == 0 || height == 0) {
    Log.fatal("Invalid render texture size! {}x{}, multisampled {}", width, height, multisampled);
  }
  const auto format = g_graphicsConfig.surfaceFormat;
  auto texture = std::shared_ptr<rhi::Texture>{g_device->create_texture(
      {width, height, 1, multisampled ? g_graphicsConfig.msaaSamples : 1, format,
       rhi::TextureSampled | rhi::TextureColorTarget})};
  std::shared_ptr<rhi::TextureView> view = texture->create_view({});
  return {
      .texture = std::move(texture),
      .view = std::move(view),
      .size = {width, height},
      .format = format,
      .sampler = create_clamped_sampler(),
  };
}

const TextureWithSampler& present_source() noexcept {
  return g_graphicsConfig.msaaSamples > 1 ? g_frameBufferResolved : g_frameBuffer;
}

Viewport calculate_present_viewport(uint32_t surface_width, uint32_t surface_height, uint32_t content_width,
                                    uint32_t content_height) noexcept {
  if (surface_width == 0 || surface_height == 0 || content_width == 0 || content_height == 0) {
    return {};
  }

  uint32_t viewport_width = surface_width;
  uint32_t viewport_height = std::min<uint32_t>(
      surface_height, std::max<uint32_t>(1u, static_cast<uint32_t>(std::lround(static_cast<double>(viewport_width) *
                                                                               static_cast<double>(content_height) /
                                                                               static_cast<double>(content_width)))));
  if (viewport_height == surface_height) {
    viewport_width = std::min<uint32_t>(
        surface_width, std::max<uint32_t>(1u, static_cast<uint32_t>(std::lround(static_cast<double>(viewport_height) *
                                                                                static_cast<double>(content_width) /
                                                                                static_cast<double>(content_height)))));
  }

  return {
      .left = static_cast<float>((surface_width - viewport_width) / 2),
      .top = static_cast<float>((surface_height - viewport_height) / 2),
      .width = static_cast<float>(viewport_width),
      .height = static_cast<float>(viewport_height),
      .znear = 0.f,
      .zfar = 1.f,
  };
}

static TextureWithSampler create_depth_texture(uint32_t width, uint32_t height) {
  const auto format = g_graphicsConfig.depthFormat;
  auto texture = std::shared_ptr<rhi::Texture>{g_device->create_texture(
      {width, height, 1, g_graphicsConfig.msaaSamples, format, rhi::TextureSampled | rhi::TextureDepthTarget})};
  std::shared_ptr<rhi::TextureView> view = texture->create_view({});
  return {
      .texture = std::move(texture),
      .view = std::move(view),
      .size = {width, height},
      .format = format,
      .sampler = create_clamped_sampler(),
  };
}

std::unique_ptr<rhi::BindGroup> create_copy_bind_group(const TextureWithSampler& source) {
  std::array<rhi::BindingValue, 2> entries{};
  entries[0].binding = 0;
  entries[0].sampler = source.sampler.get();
  entries[1].binding = 1;
  entries[1].view = source.view.get();
  return g_device->create_bind_group({g_CopyBindGroupLayout.get(), entries.data(), entries.size()});
}

const TextureWithSampler& resample_present_source(rhi::CommandList& list, const Viewport& viewport) {
  const auto& source = present_source();
  const uint32_t width = viewport_extent(viewport.width);
  const uint32_t height = viewport_extent(viewport.height);
  // A frame already at the window's size goes straight to the present copy.
  if (source.size.width == width && source.size.height == height) {
    return source;
  }
  if (!g_resampledFrameBuffer.view || g_resampledFrameBuffer.size.width != width ||
      g_resampledFrameBuffer.size.height != height || g_resampledFrameBuffer.format != source.format) {
    g_resampledFrameBuffer = create_render_texture(width, height, false);
  }

  const ResampleUniformBlock uniform{
      // Rendered above the window (render scale): an area filter averages
      // every pixel each output pixel covers; bilinear only reads four.
      .samplerMode = sampler_mode(source.size.width > width || source.size.height > height ? SAMPLER_AREA
                                                                                           : SAMPLER_BILINEAR),
      .frameWidth = static_cast<float>(width),
      .frameHeight = static_cast<float>(height),
  };
  ASSERT(gfx::render_worker::is_worker_thread(), "Present resample must run on the render worker");
  static uint32_t slot = 0;
  slot = (slot + 1) % ResampleUniformSlots;
  const uint32_t offset = slot * ResampleUniformStride;
  std::memcpy(static_cast<uint8_t*>(g_ResampleUniformBuffer->mapped()) + offset, &uniform, sizeof(uniform));

  std::array<rhi::BindingValue, 3> entries{};
  entries[0].binding = 0;
  entries[0].buffer = g_ResampleUniformBuffer.get();
  entries[0].size = ResampleUniformStride;
  entries[1].binding = 1;
  entries[1].sampler = source.sampler.get();
  entries[2].binding = 2;
  entries[2].view = source.view.get();
  const auto bindGroup = g_device->create_bind_group({g_ResampleBindGroupLayout.get(), entries.data(), entries.size()});

  rhi::RenderPassDesc pass;
  pass.color.target = g_resampledFrameBuffer.texture.get();
  pass.color.clear = true;
  list.begin_render_pass(pass);
  list.set_pipeline(*g_ResamplePipeline);
  list.set_bind_group(0, *bindGroup, &offset, 1);
  list.draw(3, 0);
  list.end_render_pass();

  return g_resampledFrameBuffer;
}

static void release_surface_locked() noexcept {
  g_surface = nullptr;
  s_surface.reset();
}

static bool create_surface() {
  release_surface_locked();
  auto* window = window::get_sdl_window();
  if (window == nullptr) {
    return false;
  }
  const SDL_PropertiesID properties = SDL_GetWindowProperties(window);
  rhi::SwapchainDesc desc;
  desc.window.window = SDL_GetPointerProperty(properties, SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr);
  desc.window.instance = SDL_GetPointerProperty(properties, SDL_PROP_WINDOW_WIN32_INSTANCE_POINTER, nullptr);
  desc.width = g_graphicsConfig.surfaceWidth;
  desc.height = g_graphicsConfig.surfaceHeight;
  desc.format = g_graphicsConfig.surfaceFormat;
  desc.mode = g_graphicsConfig.presentMode;
  s_surface = g_device->create_swapchain(desc);
  g_surface = s_surface.get();
  if (g_surface == nullptr) {
    Log.error("Failed to create the swapchain");
    return false;
  }
  return true;
}

bool initialize(AuroraBackend auroraBackend, bool allowCpu) {
  (void)allowCpu;
  const rhi::Backend backend = auroraBackend == BACKEND_VULKAN ? rhi::Backend::Vulkan : rhi::Backend::D3D12;
  rhi::DeviceOptions options;
  options.validation = std::getenv("RHI_VALIDATION") != nullptr;
  std::string error;
  s_device = rhi::create_device(backend, options, &error);
  if (!s_device) {
    Log.error("Failed to create the device: {}", error);
    return false;
  }
  g_device = s_device.get();
  Log.info("Graphics adapter: {} ({})", g_device->adapter_name(), backend == rhi::Backend::Vulkan ? "Vulkan" : "D3D12");

  const auto size = window::get_window_size();
  g_graphicsConfig = GraphicsConfig{
      .surfaceFormat = rhi::Format::BGRA8Unorm,
      .surfaceWidth = size.native_fb_width,
      .surfaceHeight = size.native_fb_height,
      .presentMode = best_present_mode(g_config.vsync),
      .depthFormat = rhi::Format::Depth32Float,
      .msaaSamples = std::min(g_config.msaa, g_device->features().max_samples),
      .textureAnisotropy = g_config.maxTextureAnisotropy,
  };
  if (g_graphicsConfig.msaaSamples != g_config.msaa) {
    Log.warn("MSAA {}x asked, the GPU does {}x", g_config.msaa, g_graphicsConfig.msaaSamples);
  }
  if (!create_surface()) {
    return false;
  }
  create_copy_pipeline();
  create_resample_pipeline();
  resize_swapchain(size.fb_width, size.fb_height, size.native_fb_width, size.native_fb_height, true);
  return true;
}

void shutdown() {
  gfx::gpu_synchronize();
  if (g_device != nullptr) {
    g_device->wait_idle();
  }
  g_CopyPipeline = nullptr;
  s_CopyPipeline.reset();
  g_CopyPipelineLayout.reset();
  g_CopyBindGroupLayout.reset();
  g_ResamplePipeline.reset();
  g_ResamplePipelineLayout.reset();
  g_ResampleBindGroupLayout.reset();
  g_ResampleUniformBuffer.reset();
  g_resampledFrameBuffer = {};
  g_frameBuffer = {};
  g_frameBufferResolved = {};
  g_depthBuffer = {};
  release_surface_locked();
  // Stays for the process's last moments: textures in static state go away at exit,
  // and retire with their device.
  (void)s_device.release();
}

void release_surface() noexcept {
  gfx::gpu_synchronize();
  if (g_device != nullptr) {
    g_device->wait_idle();
  }
  release_surface_locked();
}

static void resize_swapchain_internal(uint32_t width, uint32_t height, uint32_t nativeWidth, uint32_t nativeHeight,
                                      bool force) {
  if (g_surface == nullptr || g_device == nullptr || width == 0 || height == 0 || nativeHeight == 0 ||
      nativeWidth == 0) {
    return;
  }
  const bool surfaceChanged =
      g_graphicsConfig.surfaceWidth != nativeWidth || g_graphicsConfig.surfaceHeight != nativeHeight;
  const bool sizeChanged = surfaceChanged || g_frameBuffer.size.width != width || g_frameBuffer.size.height != height;
  if (!force && !sizeChanged) {
    return;
  }
  if (sizeChanged) {
    // Replays draw into the render targets replaced below.
    gfx::interp::discard();
    gfx::clear_caches();
  }
  g_graphicsConfig.surfaceWidth = nativeWidth;
  g_graphicsConfig.surfaceHeight = nativeHeight;
  if (surfaceChanged) {
    g_surface->resize(nativeWidth, nativeHeight);
  }
  g_frameBuffer = create_render_texture(width, height, true);
  g_frameBufferResolved = create_render_texture(width, height, false);
  g_depthBuffer = create_depth_texture(width, height);
}

bool refresh_surface(bool recreate) {
  gfx::gpu_synchronize();
  if (g_device == nullptr) {
    return false;
  }
  g_device->wait_idle();
  if (!window::is_presentable()) {
    release_surface_locked();
    return false;
  }
  if ((g_surface == nullptr || recreate) && !create_surface()) {
    return false;
  }
  uint32_t width = g_graphicsConfig.surfaceWidth;
  uint32_t height = g_graphicsConfig.surfaceHeight;
  uint32_t nativeWidth = width;
  uint32_t nativeHeight = height;
  if (window::get_sdl_window() != nullptr) {
    const auto size = window::get_window_size();
    width = size.fb_width;
    height = size.fb_height;
    nativeWidth = size.native_fb_width;
    nativeHeight = size.native_fb_height;
  }
  if (width != 0 && height != 0) {
    resize_swapchain_internal(width, height, nativeWidth, nativeHeight, true);
  }
  return true;
}

void resize_swapchain(uint32_t width, uint32_t height, uint32_t nativeWidth, uint32_t nativeHeight, bool force) {
  gfx::gpu_synchronize();
  resize_swapchain_internal(width, height, nativeWidth, nativeHeight, force);
}
std::vector<std::string> shader_sources() {
  return {std::string{FullscreenVertex} + CopyShader, std::string{FullscreenVertex} + ResampleShader};
}

} // namespace aurora::gpu
