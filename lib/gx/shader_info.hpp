#pragma once

#include "gx.hpp"

namespace aurora::gx {
ShaderInfo build_shader_info(const ShaderConfig& config) noexcept;
// layout receives where the block's parts frame interpolation blends sit.
gfx::Range build_uniform(const ShaderInfo& info, uint32_t vtxStart, const BindGroupRanges& ranges,
                         gfx::interp::BlockLayout* layout) noexcept;
u8 color_channel(GXChannelID id) noexcept;
}; // namespace aurora::gx
