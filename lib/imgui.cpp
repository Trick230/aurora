#include "imgui.hpp"

#include <cstddef>
#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <SDL3/SDL_events.h>

#include "internal.hpp"
#include "gfx/render_worker.hpp"
#include "gfx/texture.hpp"
#include "gpu/gpu.hpp"
#include "window.hpp"

#include "backends/imgui_impl_sdl3.h"

namespace aurora::imgui {
static std::string g_imguiSettings{};
static std::string g_imguiLog{};
// Off until the game opens a UI: with nothing to show, ImGui does no work at all.
static bool g_active = false;
// Whether this frame called ImGui::NewFrame, so freeze pairs with it however
// g_active changes in between.
static bool g_frameStarted = false;


using gpu::g_device;

static_assert(sizeof(ImDrawIdx) == 2, "ImGui draws with 16-bit indices");
static_assert(sizeof(ImDrawVert) == 20, "the shader reads ImDrawVert by hand");

constexpr const char* Shader = R"(
struct Output {
  float4 pos : SV_Position;
  float2 uv : TEXCOORD0;
  float4 color : COLOR0;
};
ByteAddressBuffer vertices : register(t0, space0);
SamplerState s : register(s0, space1);
Texture2D t : register(t1, space1);

Output vs_main(uint id : SV_VertexID) {
  const uint at = id * 20;
  const uint color = vertices.Load(at + 16);
  Output output;
  output.pos = float4(asfloat(vertices.Load2(at)), 0.0, 1.0);
  output.uv = asfloat(vertices.Load2(at + 8));
  output.color = float4(color & 255u, (color >> 8) & 255u, (color >> 16) & 255u, color >> 24) / 255.0;
  return output;
}

float4 fs_main(Output input) : SV_Target {
  return input.color * t.Sample(s, input.uv);
}
)";

// A texture ImGui asked for; its ImTextureID points to one of these.
struct Texture {
  std::shared_ptr<rhi::Texture> texture;
  std::unique_ptr<rhi::TextureView> view;
  std::unique_ptr<rhi::BindGroup> group; // the render worker's
};

// All of it the render worker's.
static std::unique_ptr<rhi::BindGroupLayout> g_vertexLayout;
static std::unique_ptr<rhi::BindGroupLayout> g_textureLayout;
static std::unique_ptr<rhi::PipelineLayout> g_pipelineLayout;
static std::unique_ptr<rhi::RenderPipeline> g_pipeline;
static std::unique_ptr<rhi::Sampler> g_sampler;

static void create_pipeline() {
  const rhi::BindingLayout vertex{0, rhi::BindingKind::StorageRead};
  const rhi::BindingLayout texture[] = {{0, rhi::BindingKind::Sampler}, {1, rhi::BindingKind::Texture}};
  g_vertexLayout = g_device->create_bind_group_layout({&vertex, 1});
  g_textureLayout = g_device->create_bind_group_layout({texture, 2});
  rhi::BindGroupLayout* groups[] = {g_vertexLayout.get(), g_textureLayout.get()};
  g_pipelineLayout = g_device->create_pipeline_layout({groups, 2});
  const auto vs = gpu::compile_shader(rhi::ShaderStage::Vertex, Shader, "vs_main");
  const auto fs = gpu::compile_shader(rhi::ShaderStage::Fragment, Shader, "fs_main");
  rhi::RenderPipelineDesc desc;
  desc.layout = g_pipelineLayout.get();
  desc.vertex = &vs;
  desc.vertex_entry = "vs_main";
  desc.fragment = &fs;
  desc.fragment_entry = "fs_main";
  desc.color_format = gpu::g_graphicsConfig.surfaceFormat;
  desc.blend = true;
  desc.blend_color = {rhi::BlendFactor::SrcAlpha, rhi::BlendFactor::OneMinusSrcAlpha};
  desc.blend_alpha = {rhi::BlendFactor::One, rhi::BlendFactor::OneMinusSrcAlpha};
  g_pipeline = g_device->create_render_pipeline(desc);
  rhi::SamplerDesc sampler;
  sampler.u = rhi::AddressMode::ClampToEdge;
  sampler.v = rhi::AddressMode::ClampToEdge;
  g_sampler = g_device->create_sampler(sampler);
}

struct DrawData::Impl {
  ImDrawData drawData;
  std::vector<std::unique_ptr<ImDrawList>> drawLists;
  // What the draws sample stays alive for as long as they may be replayed.
  std::shared_ptr<Texture> fonts;
};

// The font atlas, and whether its pixels still have to go up with a frame.
static std::shared_ptr<Texture> g_fonts;
static bool g_fontsPending = false;

static void create_fonts() {
  unsigned char* pixels = nullptr;
  int width = 0, height = 0;
  ImGuiIO& io = ImGui::GetIO();
  io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
  g_fonts = std::make_shared<Texture>();
  g_fonts->texture = g_device->create_texture(
      {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1, 1, rhi::Format::RGBA8Unorm, rhi::TextureSampled});
  g_fonts->view = g_fonts->texture->create_view({});
  io.Fonts->SetTexID(reinterpret_cast<ImTextureID>(g_fonts.get()));
  g_fontsPending = true;
}

static void upload_fonts() {
  unsigned char* pixels = nullptr;
  int width = 0, height = 0;
  ImGui::GetIO().Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
  gfx::queue_texture_upload_data(pixels, static_cast<uint32_t>(width) * 4, static_cast<uint32_t>(height),
                                 g_fonts->texture, 0, {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1});
  g_fontsPending = false;
}

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
  ImGuiIO& io = ImGui::GetIO();
  io.BackendRendererName = "aurora_rhi";
  io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;
}

bool active() noexcept { return g_active; }
void set_active(bool active) noexcept { g_active = active; }

void shutdown() noexcept {
  ImGui_ImplSDL3_Shutdown();
  ImGui::DestroyContext();
  g_fonts.reset();
  g_sampler.reset();
  g_pipeline.reset();
  g_pipelineLayout.reset();
  g_textureLayout.reset();
  g_vertexLayout.reset();
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
  ImGui_ImplSDL3_NewFrame();
  ImGuiIO& io = ImGui::GetIO();
  if (!g_fonts || !io.Fonts->IsBuilt()) {
    create_fonts();
  }
  io.DisplayFramebufferScale = {
      size.width > 0 ? static_cast<float>(size.native_fb_width) / static_cast<float>(size.width) : 1.f,
      size.height > 0 ? static_cast<float>(size.native_fb_height) / static_cast<float>(size.height) : 1.f,
  };
  io.DisplaySize = {static_cast<float>(size.width), static_cast<float>(size.height)};
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
  if (g_fontsPending) {
    upload_fonts();
  }
  auto frozen = std::make_shared<DrawData::Impl>();
  frozen->fonts = g_fonts;
  frozen->drawData = *data;
  frozen->drawLists.reserve(data->CmdListsCount);
  frozen->drawData.CmdLists.resize(data->CmdListsCount);
  for (int i = 0; i < data->CmdListsCount; ++i) {
    frozen->drawLists.emplace_back(data->CmdLists[i]->CloneOutput());
    frozen->drawData.CmdLists[i] = frozen->drawLists.back().get();
  }
  return DrawData{std::move(frozen)};
}

void render(rhi::CommandList& list, rhi::Texture& target, const DrawData& drawData) noexcept {
  if (!drawData) {
    return;
  }
  const ImDrawData& data = drawData.m_impl->drawData;
  if (data.TotalVtxCount == 0 || data.TotalIdxCount == 0) {
    return;
  }
  if (!g_pipeline) {
    create_pipeline();
  }

  // Both go when this recording is submitted, as the RHI lets what the next submit uses.
  const auto vertices = g_device->create_buffer({data.TotalVtxCount * sizeof(ImDrawVert), rhi::MemoryKind::Upload});
  const auto indices = g_device->create_buffer({data.TotalIdxCount * sizeof(ImDrawIdx), rhi::MemoryKind::Upload});
  auto* vertexOut = static_cast<ImDrawVert*>(vertices->mapped());
  auto* indexOut = static_cast<ImDrawIdx*>(indices->mapped());
  // Clip space goes straight in the vertices, so nothing else needs uploading.
  const float scaleX = 2.f / data.DisplaySize.x, scaleY = -2.f / data.DisplaySize.y;
  for (const ImDrawList* const drawList : data.CmdLists) {
    for (const ImDrawVert& vertex : drawList->VtxBuffer) {
      *vertexOut = vertex;
      vertexOut->pos.x = (vertex.pos.x - data.DisplayPos.x) * scaleX - 1.f;
      vertexOut->pos.y = (vertex.pos.y - data.DisplayPos.y) * scaleY + 1.f;
      ++vertexOut;
    }
    std::memcpy(indexOut, drawList->IdxBuffer.Data, drawList->IdxBuffer.Size * sizeof(ImDrawIdx));
    indexOut += drawList->IdxBuffer.Size;
  }
  rhi::BindingValue vertexValue{};
  vertexValue.buffer = vertices.get();
  vertexValue.size = data.TotalVtxCount * sizeof(ImDrawVert);
  const auto vertexGroup = g_device->create_bind_group({g_vertexLayout.get(), &vertexValue, 1});

  const float width = data.DisplaySize.x * data.FramebufferScale.x;
  const float height = data.DisplaySize.y * data.FramebufferScale.y;
  rhi::RenderPassDesc pass;
  pass.color.target = &target;
  list.begin_render_pass(pass);
  list.set_pipeline(*g_pipeline);
  list.set_viewport(0.f, 0.f, width, height, 0.f, 1.f);
  list.set_index_buffer_u16(*indices, 0);
  list.set_bind_group(0, *vertexGroup, nullptr, 0);
  int firstVertex = 0;
  uint32_t firstIndex = 0;
  for (const ImDrawList* const drawList : data.CmdLists) {
    for (const ImDrawCmd& cmd : drawList->CmdBuffer) {
      if (cmd.UserCallback != nullptr) {
        if (cmd.UserCallback != ImDrawCallback_ResetRenderState) {
          cmd.UserCallback(drawList, &cmd);
        }
        continue;
      }
      const float left = std::max((cmd.ClipRect.x - data.DisplayPos.x) * data.FramebufferScale.x, 0.f);
      const float top = std::max((cmd.ClipRect.y - data.DisplayPos.y) * data.FramebufferScale.y, 0.f);
      const float right = std::min((cmd.ClipRect.z - data.DisplayPos.x) * data.FramebufferScale.x, width);
      const float bottom = std::min((cmd.ClipRect.w - data.DisplayPos.y) * data.FramebufferScale.y, height);
      if (right <= left || bottom <= top) {
        continue;
      }
      list.set_scissor(static_cast<uint32_t>(left), static_cast<uint32_t>(top), static_cast<uint32_t>(right - left),
                       static_cast<uint32_t>(bottom - top));
      auto* const texture = reinterpret_cast<Texture*>(cmd.GetTexID());
      if (!texture->group) {
        rhi::BindingValue values[2]{};
        values[0].binding = 0;
        values[0].sampler = g_sampler.get();
        values[1].binding = 1;
        values[1].view = texture->view.get();
        texture->group = g_device->create_bind_group({g_textureLayout.get(), values, 2});
      }
      list.set_bind_group(1, *texture->group, nullptr, 0);
      list.draw_indexed(cmd.ElemCount, firstIndex + cmd.IdxOffset, firstVertex + static_cast<int>(cmd.VtxOffset), 1);
    }
    firstVertex += drawList->VtxBuffer.Size;
    firstIndex += drawList->IdxBuffer.Size;
  }
  list.end_render_pass();
}

std::vector<std::string> shader_sources() { return {Shader}; }

} // namespace aurora::imgui

// C bindings
extern "C" {
void aurora_imgui_set_active(bool active) { aurora::imgui::set_active(active); }
bool aurora_imgui_active() { return aurora::imgui::active(); }
}
