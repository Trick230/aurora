#ifndef AURORA_IMGUI_H
#define AURORA_IMGUI_H

#include <imgui.h>

#ifdef __cplusplus
#include <cstdint>

extern "C" {
#else
#include "stdint.h"
#endif

// ImGui runs only while active: set it when a menu opens, clear it when it closes.
void aurora_imgui_set_active(bool active);
bool aurora_imgui_active();
// A texture for the host's ImGui draws, from width*height RGBA8 pixels (copied).
// Call on the thread that draws; the texture stays until shutdown.
ImTextureID aurora_imgui_add_texture(const uint8_t* rgba, uint32_t width, uint32_t height);

#ifdef __cplusplus
}
#endif

#endif
