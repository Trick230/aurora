#pragma once

#include "gx.hpp"

#include <string>

namespace aurora::gx::hlsl {
// One source with vs_main and fs_main, for rhi::compile_shader. Group 0 holds the
// vertex data (t0, t1), group 1 the uniform (b0, with a dynamic offset), group 2 the
// textures (t and s, two bindings each), and the draw's vertex offset is its one
// draw constant.
std::string build_shader_source(const ShaderConfig& config, DstAlphaMode dstAlphaMode) noexcept;
} // namespace aurora::gx::hlsl
