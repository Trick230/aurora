#pragma once

#include <aurora/event.h>

#include "rhi.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

union SDL_Event;

namespace aurora::imgui {
class DrawData {
public:
  DrawData() noexcept = default;
  [[nodiscard]] explicit operator bool() const noexcept { return m_impl != nullptr; }

private:
  struct Impl;
  explicit DrawData(std::shared_ptr<Impl> impl) noexcept : m_impl(std::move(impl)) {}

  std::shared_ptr<Impl> m_impl;

  friend DrawData freeze() noexcept;
  friend void render(rhi::CommandList& list, rhi::Texture& target, const DrawData& drawData) noexcept;
};

void create_context() noexcept;
void initialize() noexcept;
void shutdown() noexcept;
// Whether a UI is up. While it isn't, frames skip ImGui and its render pass.
bool active() noexcept;
void set_active(bool active) noexcept;

void process_event(const SDL_Event& event) noexcept;
void new_frame(const AuroraWindowSize& size) noexcept;
DrawData freeze() noexcept;
void render(rhi::CommandList& list, rhi::Texture& target, const DrawData& drawData) noexcept;
// The HLSL sources of its pipelines, for shipping them compiled.
std::vector<std::string> shader_sources();
} // namespace aurora::imgui
