#pragma once

#include <aurora/aurora.h>
#include <aurora/event.h>

struct SDL_Window;

namespace aurora::window {

enum class CustomEvent {
  RefreshSurface,
};
inline Uint32 operator+(Uint32 lhs, CustomEvent rhs) { return lhs + static_cast<Uint32>(rhs); }

bool initialize();
bool initialize_event_watch();
bool push_custom_event(CustomEvent eventType);
void shutdown();
bool create_window();
void destroy_window();
void show_window();
AuroraWindowSize get_window_size();
const AuroraEvent* poll_events();
SDL_Window* get_sdl_window();
bool is_paused() noexcept;
bool is_presentable() noexcept;
// Between begin_frame and end_frame the frame being recorded uses the render
// targets and caches a resize replaces: resizes wait for the frame to end.
void set_frame_open(bool open) noexcept;
}; // namespace aurora::window
