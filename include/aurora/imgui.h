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

#ifdef __cplusplus
}
#endif

#endif
