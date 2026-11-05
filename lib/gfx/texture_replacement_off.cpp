// Texture replacement is not ported to the RHI yet.
#include "texture_replacement.hpp"

#include <aurora/texture.hpp>

namespace aurora::gfx::texture_replacement {
void initialize() noexcept {}
void shutdown() noexcept {}
std::optional<TextureHandle> find_replacement(const GXTexObj_&) noexcept { return std::nullopt; }
std::optional<TextureHandle> find_replacement(const GXTexObj_&, const GXTlutObj_&) noexcept { return std::nullopt; }
std::string build_texture_replacement_name(const GXTexObj_&) noexcept { return {}; }
} // namespace aurora::gfx::texture_replacement

namespace aurora::texture {
void load_replacement_directory(const std::filesystem::path&) {}
} // namespace aurora::texture
