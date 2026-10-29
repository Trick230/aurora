#ifndef AURORA_TEXTURE_HPP
#define AURORA_TEXTURE_HPP

#include <cstdint>
#include <filesystem>

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
} // namespace aurora::texture

#endif
