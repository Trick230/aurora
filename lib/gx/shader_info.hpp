#pragma once

#include "gx.hpp"

namespace aurora::gx {
ShaderInfo build_shader_info(const ShaderConfig& config) noexcept;
// layout receives where the block's parts frame interpolation blends sit.
gfx::Range build_uniform(const ShaderInfo& info, uint32_t vtxStart, const BindGroupRanges& ranges,
                         gfx::interp::BlockLayout* layout) noexcept;
// The ubershader's block: everything a shader may read, in one layout.
// configBase is push_uber_config's result.
gfx::Range build_uber_uniform(const ShaderInfo& info, uint32_t vtxStart, const BindGroupRanges& ranges,
                              uint32_t configBase, gfx::interp::BlockLayout* layout) noexcept;
u8 color_channel(GXChannelID id) noexcept;
f32 tex_offset_value(GXTexOffset offs) noexcept;
u32 point_texcoord_mask() noexcept;
u32 line_texcoord_mask() noexcept;
// What the shader generator and the ubershader decide alike.
bool uses_texture_sample(const TevStage& stage) noexcept;
bool tev_color_arg_is_normalized(GXTevColorArg arg, const std::array<bool, MaxTevRegs>& colorNormalized,
                                 const std::array<bool, MaxTevRegs>& alphaNormalized);
bool tev_alpha_arg_is_normalized(GXTevAlphaArg arg, const std::array<bool, MaxTevRegs>& alphaNormalized);
bool tev_alpha_compares_color(GXTevOp op);
// The vertex fetch and TEV helpers, as WGSL.
std::string shader_helpers() noexcept;
}; // namespace aurora::gx
