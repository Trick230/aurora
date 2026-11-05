#pragma once

#include "rhi.hpp"

#include <string>

namespace aurora::gpu::shader_cache {
// Compiled shaders by their source: what shipped with the game
// (<resources>/shaders_<backend>.bin) and what this machine compiled since
// (<cache>/shaders_<backend>.bin). Either is dropped when the compiler or its
// arguments change. Thread-safe.
bool find(rhi::ShaderStage stage, const std::string& source, const char* entry, rhi::ShaderCode* code);
void store(rhi::ShaderStage stage, const std::string& source, const char* entry, const rhi::ShaderCode& code);
} // namespace aurora::gpu::shader_cache
