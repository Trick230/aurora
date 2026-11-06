#pragma once

#include "rhi.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace aurora::gpu::shader_cache {
// Compiled shaders by their source: what shipped with the game
// (<resources>/shaders_<backend>.bin) and what this machine compiled since
// (<cache>/shaders_<backend>.bin). Either is dropped when the compiler or its
// arguments change. Thread-safe.
bool find(rhi::ShaderStage stage, const std::string& source, const char* entry, rhi::ShaderCode* code);
void store(rhi::ShaderStage stage, const std::string& source, const char* entry, const rhi::ShaderCode& code);

// For tools that make the shipped files.
struct Key {
  uint64_t low, high;
  bool operator==(const Key&) const = default;
  template <typename H>
  friend H AbslHashValue(H state, const Key& key) {
    return H::combine(std::move(state), key.low, key.high);
  }
};
using Records = std::vector<std::pair<Key, rhi::ShaderCode>>;
Key key(rhi::ShaderStage stage, const std::string& source, const char* entry);
const char* file_name(rhi::Backend backend);
// What a file is only good for: the backend and the compiler.
std::string salt(rhi::Backend backend);
// A file's records, if it is for this salt.
bool read_file(const std::filesystem::path& path, const std::string& salt, Records* records);
bool write_file(const std::filesystem::path& path, const std::string& salt, const Records& records);
} // namespace aurora::gpu::shader_cache
