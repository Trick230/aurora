#include "imgui.hpp"

#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <webgpu/webgpu_cpp.h>
#include <SDL3/SDL_events.h>

#include "internal.hpp"
#include "gfx/render_worker.hpp"
#include "webgpu/gpu.hpp"
#include "window.hpp"

#define IMGUI_IMPL_WEBGPU_BACKEND_DAWN
#include "backends/imgui_impl_sdl3.h"
#include "backends/imgui_impl_wgpu.h"

namespace aurora::imgui {
static float g_scale;
static std::string g_imguiSettings{};
static std::string g_imguiLog{};
// Off until the game opens a UI: with nothing to show, ImGui does no work at all.
static bool g_active = false;
// Whether this frame called ImGui::NewFrame, so freeze pairs with it however
// g_active changes in between.
static bool g_frameStarted = false;


struct DrawData::Impl {
  ImDrawData drawData;
  std::vector<std::unique_ptr<ImDrawList>> drawLists;
};

void create_context() noexcept {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  g_imguiSettings = std::string{g_config.userPath} + "/imgui.ini";
  g_imguiLog = std::string{g_config.cachePath} + "/imgui.log";
  io.IniFilename = g_imguiSettings.c_str();
  io.LogFilename = g_imguiLog.c_str();
}

void initialize() noexcept {
  ImGui_ImplSDL3_InitForOther(window::get_sdl_window());
  ImGui_ImplWGPU_InitInfo info;
  info.Device = webgpu::g_device.Get();
  info.RenderTargetFormat = static_cast<WGPUTextureFormat>(webgpu::g_graphicsConfig.surfaceConfiguration.format);
  ImGui_ImplWGPU_Init(&info);
}

bool active() noexcept { return g_active; }
void set_active(bool active) noexcept { g_active = active; }

void shutdown() noexcept {
  ImGui_ImplWGPU_Shutdown();
  ImGui_ImplSDL3_Shutdown();
  ImGui::DestroyContext();
}

void process_event(const SDL_Event& event) noexcept {
  if (g_active) {
    ImGui_ImplSDL3_ProcessEvent(&event);
  }
}

void new_frame(const AuroraWindowSize& size) noexcept {
  g_frameStarted = g_active;
  if (!g_frameStarted) {
    return;
  }
  ImVec2 framebufferScale{
      size.width > 0 ? static_cast<float>(size.native_fb_width) / static_cast<float>(size.width) : 1.0f,
      size.height > 0 ? static_cast<float>(size.native_fb_height) / static_cast<float>(size.height) : 1.0f,
  };
  const ImVec2 displaySize{static_cast<float>(size.width), static_cast<float>(size.height)};

  if (g_scale != size.scale) {
    if (g_scale > 0.f) {
      ImGui_ImplWGPU_CreateDeviceObjects();
    }
    g_scale = size.scale;
  }
  if (!ImGui::GetIO().Fonts->IsBuilt()) {
    ImGui_ImplWGPU_CreateDeviceObjects();
  }
  ImGui_ImplWGPU_NewFrame();
  ImGui_ImplSDL3_NewFrame();

  ImGuiIO& io = ImGui::GetIO();
  io.DisplayFramebufferScale = framebufferScale;
  ImGui::GetIO().DisplaySize = displaySize;
  ImGui::NewFrame();
}

DrawData freeze() noexcept {
  if (!g_frameStarted) {
    return {};
  }
  g_frameStarted = false;
  ImGui::Render();

  auto* data = ImGui::GetDrawData();
  data->FramebufferScale = ImGui::GetIO().DisplayFramebufferScale;
  auto frozen = std::make_shared<DrawData::Impl>();
  frozen->drawData = *data;
  frozen->drawLists.reserve(data->CmdListsCount);
  frozen->drawData.CmdLists.resize(data->CmdListsCount);
  for (int i = 0; i < data->CmdListsCount; ++i) {
    frozen->drawLists.emplace_back(data->CmdLists[i]->CloneOutput());
    frozen->drawData.CmdLists[i] = frozen->drawLists.back().get();
  }
  return DrawData{std::move(frozen)};
}

void render(const wgpu::RenderPassEncoder& pass, const DrawData& drawData) noexcept {

  if (!drawData.m_impl) {
    return;
  }
  auto* data = &drawData.m_impl->drawData;
  if (data->CmdListsCount == 0) {
    return;
  }
  pass.PushDebugGroup("Aurora: Dear Imgui");
  ImGui_ImplWGPU_RenderDrawData(data, pass.Get());
  pass.PopDebugGroup();
}

} // namespace aurora::imgui

// C bindings
extern "C" {
void aurora_imgui_set_active(bool active) { aurora::imgui::set_active(active); }
bool aurora_imgui_active() { return aurora::imgui::active(); }
}
