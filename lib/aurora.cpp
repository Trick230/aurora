#include <aurora/aurora.h>

#include "gfx/common.hpp"
#include "gfx/render_worker.hpp"
#include "imgui.hpp"
#include "webgpu/gpu.hpp"

#include "internal.hpp"
#include "window.hpp"

#include <SDL3/SDL_filesystem.h>
#include <magic_enum.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>


namespace aurora {
AuroraConfig g_config;
uint32_t g_sdlCustomEventsStart;
char g_gameName[4];

namespace {
Module Log("aurora");

// GPU
using webgpu::g_device;
using webgpu::g_surface;

uint32_t clamp_scissor_coord(double value, uint32_t maximum) noexcept {
  if (!std::isfinite(value)) {
    return 0;
  }
  return static_cast<uint32_t>(std::clamp(value, 0.0, static_cast<double>(maximum)));
}

// AURORA_FRAME_DUMP=<dir>: each frame's hash goes to hashes.txt there, and every
// 30th frame's RGBA pixels to <frame>.raw, to compare renderers by numbers.
const char* frame_dump_dir() noexcept {
  static const char* dir = std::getenv("AURORA_FRAME_DUMP");
  return dir;
}

struct FrameDump {
  std::unique_ptr<rhi::Buffer> buffer;
  uint32_t width = 0, height = 0, row = 0;
  bool bgra = false;
};

FrameDump frame_dump_copy(rhi::CommandList& list) {
  const auto& source = webgpu::present_source();
  FrameDump dump{.width = source.size.width, .height = source.size.height};
  dump.row = AURORA_ALIGN(dump.width * 4, 256);
  dump.bgra = source.format == rhi::Format::BGRA8Unorm;
  dump.buffer = g_device->create_buffer({uint64_t{dump.row} * dump.height, rhi::MemoryKind::Readback});
  list.copy_texture_to_buffer(*source.texture, {}, *dump.buffer, 0, dump.row);
  return dump;
}

void frame_dump_write(const FrameDump& dump) {
  static uint64_t frame = 0;
  const auto* mapped = static_cast<const uint8_t*>(dump.buffer->mapped());
  std::vector<uint8_t> pixels(size_t{dump.width} * 4 * dump.height);
  for (uint32_t y = 0; y < dump.height; ++y) {
    const uint8_t* from = mapped + size_t{y} * dump.row;
    uint8_t* to = pixels.data() + size_t{y} * dump.width * 4;
    for (uint32_t x = 0; x < dump.width; ++x, from += 4, to += 4) {
      to[0] = from[dump.bgra ? 2 : 0];
      to[1] = from[1];
      to[2] = from[dump.bgra ? 0 : 2];
      to[3] = 255; // never shown, and a few pixels of it vary between runs
    }
  }
  const std::string dir = frame_dump_dir();
  if (FILE* hashes = std::fopen((dir + "/hashes.txt").c_str(), frame == 0 ? "w" : "a")) {
    std::fprintf(hashes, "%llu %016llx\n", static_cast<unsigned long long>(frame),
                 static_cast<unsigned long long>(XXH3_64bits(pixels.data(), pixels.size())));
    std::fclose(hashes);
  }
  if (frame % 30 == 0) {
    const std::string path = fmt::format("{}/{:06}.raw", dir, frame);
    if (FILE* raw = std::fopen(path.c_str(), "wb")) {
      std::fwrite(pixels.data(), 1, pixels.size(), raw);
      std::fclose(raw);
    }
  }
  ++frame;
}

void set_present_viewport(rhi::CommandList& pass, const gfx::Viewport& viewport, uint32_t surfaceWidth,
                          uint32_t surfaceHeight) noexcept {
  pass.set_viewport(viewport.left, viewport.top, viewport.width, viewport.height, viewport.znear, viewport.zfar);
  const auto scissorX = clamp_scissor_coord(std::floor(viewport.left), surfaceWidth);
  const auto scissorY = clamp_scissor_coord(std::floor(viewport.top), surfaceHeight);
  const auto scissorRight = clamp_scissor_coord(std::ceil(viewport.left + viewport.width), surfaceWidth);
  const auto scissorBottom = clamp_scissor_coord(std::ceil(viewport.top + viewport.height), surfaceHeight);
  pass.set_scissor(scissorX, scissorY, scissorRight - scissorX, scissorBottom - scissorY);
}

constexpr std::array PreferredBackendOrder{
    BACKEND_D3D12,
    BACKEND_VULKAN,
};

AuroraInfo initialize(int argc, char* argv[], const AuroraConfig& config) noexcept {
  g_config = config;
  Log.info("Aurora initializing");
  if (g_config.appName == nullptr) {
    g_config.appName = "Aurora";
  } else {
    g_config.appName = strdup(g_config.appName);
  }
  if (g_config.userPath == nullptr) {
    g_config.userPath = SDL_GetPrefPath(nullptr, g_config.appName);
  } else {
    g_config.userPath = strdup(g_config.userPath);
  }
  if (g_config.cachePath == nullptr) {
    g_config.cachePath = SDL_GetPrefPath(nullptr, g_config.appName);
  } else {
    g_config.cachePath = strdup(g_config.cachePath);
  }
  if (g_config.resourcesPath == nullptr) {
    g_config.resourcesPath = SDL_GetBasePath();
  } else {
    g_config.resourcesPath = strdup(g_config.resourcesPath);
  }
  if (g_config.msaa == 0) {
    g_config.msaa = 1;
  }
  if (g_config.maxTextureAnisotropy == 0) {
    g_config.maxTextureAnisotropy = 16;
  }
  if (!(g_config.renderScale > 0.f)) {
    g_config.renderScale = 1.f;
  }
  ASSERT(window::initialize(), "Error initializing window");

  g_sdlCustomEventsStart = SDL_RegisterEvents(1);
  ASSERT(g_sdlCustomEventsStart, "Failed to allocate user events: {}", SDL_GetError());
  ASSERT(window::initialize_event_watch(), "Error initializing SDL event watch");

  /* Attempt to create a window using the calling application's desired backend */
  AuroraBackend selectedBackend = config.desiredBackend;
  if (const char* name = std::getenv("AURORA_BACKEND")) {
    selectedBackend = std::string_view{name} == "vulkan" ? BACKEND_VULKAN : BACKEND_D3D12;
  }
  bool windowCreated = false;
  if (selectedBackend != BACKEND_AUTO && window::create_window()) {
    if (webgpu::initialize(selectedBackend, config.allowCpuAdapter)) {
      windowCreated = true;
    } else {
      window::destroy_window();
    }
  }

  if (!windowCreated) {
    for (const auto backendType : PreferredBackendOrder) {
      selectedBackend = backendType;
      if (!window::create_window()) {
        continue;
      }
      if (webgpu::initialize(selectedBackend, config.allowCpuAdapter)) {
        windowCreated = true;
        break;
      } else {
        window::destroy_window();
      }
    }
  }

  ASSERT(windowCreated, "Error creating window: {}", SDL_GetError());

  window::show_window();

  gfx::initialize();

  imgui::create_context();
  const auto size = window::get_window_size();
  Log.info("Using framebuffer size {}x{} scale {}", size.fb_width, size.fb_height, size.scale);
  imgui::initialize();

  g_config.desiredBackend = selectedBackend;
  return {
      .backend = selectedBackend,
      .userPath = g_config.userPath,
      .cachePath = g_config.cachePath,
      .window = window::get_sdl_window(),
      .windowSize = size,
  };
}

void shutdown() noexcept {
  gfx::render_worker::synchronize();
  imgui::shutdown();
  gfx::shutdown();
  webgpu::shutdown();
  window::shutdown();
}

const AuroraEvent* update() noexcept {
  return window::poll_events();
}

bool begin_frame() noexcept {
  {
    if (!window::is_presentable()) {
      webgpu::release_surface();
      return false;
    }
    if (window::is_paused()) {
      return false;
    }
    if (g_surface == nullptr) {
      webgpu::refresh_surface(true);
      if (g_surface == nullptr) {
        return false;
      }
    }
  }

  imgui::new_frame(window::get_window_size());
  if (!gfx::begin_frame()) {
    return false;
  }
  window::set_frame_open(true);
  return true;
}

void end_frame() noexcept {
  // Before finish: what ImGui uploads goes with this frame.
  auto imguiDrawData = imgui::freeze();
  gfx::finish();

  const auto& presentSource = webgpu::present_source();
  const auto viewport = webgpu::calculate_present_viewport(webgpu::g_graphicsConfig.surfaceWidth,
                                                           webgpu::g_graphicsConfig.surfaceHeight,
                                                           presentSource.size.width, presentSource.size.height);

  gfx::end_frame([viewport, imguiDrawData = std::move(imguiDrawData)](rhi::CommandList& list) -> uint64_t {
    const FrameDump frameDump = frame_dump_dir() ? frame_dump_copy(list) : FrameDump{};
    rhi::Texture* image = nullptr;
    if (window::is_presentable() && g_surface != nullptr) {
      image = g_surface->acquire();
      if (image == nullptr) {
        window::push_custom_event(window::CustomEvent::RefreshSurface);
      }
    }
    if (image != nullptr) {
      const auto& resampledSource = webgpu::resample_present_source(list, viewport);
      const auto presentBindGroup = webgpu::create_copy_bind_group(resampledSource);
      rhi::RenderPassDesc pass;
      pass.color.target = image;
      pass.color.clear = true;
      list.begin_render_pass(pass);
      // Copy EFB -> XFB (swapchain)
      list.set_pipeline(*webgpu::g_CopyPipeline);
      list.set_bind_group(0, *presentBindGroup, nullptr, 0);
      set_present_viewport(list, viewport, webgpu::g_graphicsConfig.surfaceWidth, webgpu::g_graphicsConfig.surfaceHeight);
      list.draw(3, 0);
      list.end_render_pass();
      imgui::render(list, *image, imguiDrawData);
    }
    list.end();
    const uint64_t ticket = g_device->submit(list);
    if (frameDump.buffer) {
      g_device->wait(ticket);
      frame_dump_write(frameDump);
    }
    if (image != nullptr) {
      g_surface->present();
    }
    return ticket;
  });
  window::set_frame_open(false);
}
} // namespace
} // namespace aurora

// C API bindings
AuroraInfo aurora_initialize(int argc, char* argv[], const AuroraConfig* config) {
  return aurora::initialize(argc, argv, *config);
}
void aurora_shutdown() { aurora::shutdown(); }
const AuroraEvent* aurora_update() { return aurora::update(); }
bool aurora_begin_frame() { return aurora::begin_frame(); }
void aurora_end_frame() { aurora::end_frame(); }
