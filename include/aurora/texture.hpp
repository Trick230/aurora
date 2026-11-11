#ifndef AURORA_TEXTURE_HPP
#define AURORA_TEXTURE_HPP

#include <cstdint>
#include <filesystem>
#include <span>

namespace aurora::texture {

struct TextureSourceKey {
  uint64_t textureHash = 0;
  uint64_t tlutHash = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t format = 0;
  bool hasTlut = false;

  bool operator==(const TextureSourceKey&) const = default;
};

// Registers the DDS/PNG files under root, named like Dolphin's texture dumps.
void load_replacement_directory(const std::filesystem::path& root);

// The key a GX texture (format, width, height, base level bytes) is replaced by.
TextureSourceKey source_key(uint32_t format, uint32_t width, uint32_t height, const void* data);

// Replaces the texture `key` names with RGBA8 texels the program made: the base
// level then each smaller mip, `width` by `height` at the base. Replaces the
// key's previous memory replacement; files keep their own entries.
void set_memory_replacement(const TextureSourceKey& key, uint32_t width, uint32_t height, uint32_t mips,
                            std::span<const uint8_t> rgba);
void remove_memory_replacement(const TextureSourceKey& key);
} // namespace aurora::texture

#endif
