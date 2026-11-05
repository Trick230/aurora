#pragma once

#include "gx.hpp"

#include <string>
#include <string_view>

namespace aurora::gx::hlsl {
// One source with vs_main and fs_main, for rhi::compile_shader. Group 0 holds the
// vertex data (t0, t1), group 1 the uniform (b0, with a dynamic offset), group 2 the
// textures (t and s, two bindings each), and the draw's vertex offset is its one
// draw constant.
std::string build_shader_source(const ShaderConfig& config, DstAlphaMode dstAlphaMode) noexcept;
// The vertex fetch and TEV helpers, for the generator and the ubershader.
std::string shader_helpers() noexcept;
// The ubershader, for a way destination alpha reaches the blend: the same bindings,
// with all the textures.
std::string uber_source(DstAlphaMode mode);
std::string_view uber_body() noexcept;
} // namespace aurora::gx::hlsl
