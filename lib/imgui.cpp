#include "imgui.hpp"

#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <SDL3/SDL_events.h>

#include "internal.hpp"
#include "gfx/render_worker.hpp"
#include "webgpu/gpu.hpp"
#include "window.hpp"

#define IMGUI_IMPL_WEBGPU_BACKEND_DAWN
#include "backends/imgui_impl_sdl3.h"

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
}

bool active() noexcept { return g_active; }
void set_active(bool active) noexcept { g_active = active; }

void shutdown() noexcept {
  ImGui_ImplSDL3_Shutdown();
  ImGui::DestroyContext();
}

void process_event(const SDL_Event& event) noexcept {
  if (g_active) {
    ImGui_ImplSDL3_ProcessEvent(&event);
  }
}

void new_frame(const AuroraWindowSize& size) noexcept {
  // No renderer for ImGui on the RHI yet.
  (void)size;
  g_frameStarted = false;
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

void render(rhi::CommandList&, const DrawData&) noexcept {}

} // namespace aurora::imgui

// C bindings
extern "C" {
void aurora_imgui_set_active(bool active) { aurora::imgui::set_active(active); }
bool aurora_imgui_active() { return aurora::imgui::active(); }
}
