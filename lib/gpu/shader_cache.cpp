#include "shader_cache.hpp"

#include "gpu.hpp"
#include "../internal.hpp"

#include <absl/container/flat_hash_map.h>
#include <xxhash.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <shared_mutex>

namespace aurora::gpu::shader_cache {
namespace {
Module Log("aurora::gpu::shader_cache");

// A file: this magic, the salt's length and the salt, then records of a 128-bit
// key, a 32-bit size and the code. A record cut short ends the file.
constexpr char Magic[8] = {'A', 'U', 'R', 'S', 'H', 'D', 'R', '1'};

struct Key {
  uint64_t low, high;
  bool operator==(const Key&) const = default;
  template <typename H>
  friend H AbslHashValue(H state, const Key& key) {
    return H::combine(std::move(state), key.low, key.high);
  }
};

std::shared_mutex g_mutex;
absl::flat_hash_map<Key, rhi::ShaderCode> g_shaders;
std::FILE* g_file = nullptr; // this machine's cache, open to append
bool g_loaded = false;

Key key_of(rhi::ShaderStage stage, const std::string& source, const char* entry) {
  XXH3_state_t* const state = XXH3_createState();
  XXH3_128bits_reset(state);
  const auto stageByte = static_cast<uint8_t>(stage);
  XXH3_128bits_update(state, &stageByte, 1);
  XXH3_128bits_update(state, entry, std::strlen(entry) + 1);
  XXH3_128bits_update(state, source.data(), source.size());
  const XXH128_hash_t hash = XXH3_128bits_digest(state);
  XXH3_freeState(state);
  return {hash.low64, hash.high64};
}

std::string salt() {
  return std::string{g_device->backend() == rhi::Backend::Vulkan ? "vulkan" : "d3d12"} + "|" +
         rhi::shader_compiler_version();
}

std::filesystem::path file_name() {
  return g_device->backend() == rhi::Backend::Vulkan ? "shaders_vulkan.bin" : "shaders_d3d12.bin";
}

// Reads a file's records into g_shaders; false when it isn't for this salt.
bool read(const std::filesystem::path& path, const std::string& expected, size_t* count) {
  std::FILE* const file = std::fopen(path.string().c_str(), "rb");
  if (file == nullptr) {
    return false;
  }
  char magic[sizeof(Magic)];
  uint32_t length = 0;
  std::string fileSalt;
  bool ok = std::fread(magic, 1, sizeof(magic), file) == sizeof(magic) &&
            std::memcmp(magic, Magic, sizeof(Magic)) == 0 && std::fread(&length, 4, 1, file) == 1 && length < 4096;
  if (ok) {
    fileSalt.resize(length);
    ok = std::fread(fileSalt.data(), 1, length, file) == length && fileSalt == expected;
  }
  while (ok) {
    Key key;
    uint32_t size = 0;
    if (std::fread(&key, sizeof(key), 1, file) != 1 || std::fread(&size, 4, 1, file) != 1) {
      break;
    }
    rhi::ShaderCode code(size);
    if (std::fread(code.data(), 1, size, file) != size) {
      break;
    }
    g_shaders.insert_or_assign(key, std::move(code));
    ++*count;
  }
  std::fclose(file);
  return ok;
}

// Under the exclusive lock.
void load() {
  if (g_loaded) {
    return;
  }
  g_loaded = true;
  const std::string expected = salt();
  size_t shipped = 0, local = 0;
  if (g_config.resourcesPath != nullptr && g_config.resourcesPath[0] != '\0') {
    read(std::filesystem::path{g_config.resourcesPath} / file_name(), expected, &shipped);
  }
  if (g_config.cachePath == nullptr || g_config.cachePath[0] == '\0') {
    Log.info("{} shaders shipped, no cache directory", shipped);
    return;
  }
  const auto path = std::filesystem::path{g_config.cachePath} / file_name();
  if (read(path, expected, &local)) {
    g_file = std::fopen(path.string().c_str(), "ab");
  } else {
    // Missing, or from another compiler: started over.
    g_file = std::fopen(path.string().c_str(), "wb");
    if (g_file != nullptr) {
      const auto length = static_cast<uint32_t>(expected.size());
      std::fwrite(Magic, 1, sizeof(Magic), g_file);
      std::fwrite(&length, 4, 1, g_file);
      std::fwrite(expected.data(), 1, length, g_file);
      std::fflush(g_file);
    }
  }
  Log.info("{} shaders shipped, {} compiled here before", shipped, local);
}
} // namespace

bool find(rhi::ShaderStage stage, const std::string& source, const char* entry, rhi::ShaderCode* code) {
  const Key key = key_of(stage, source, entry);
  {
    std::shared_lock lock{g_mutex};
    if (g_loaded) {
      const auto it = g_shaders.find(key);
      if (it == g_shaders.end()) {
        return false;
      }
      *code = it->second;
      return true;
    }
  }
  std::unique_lock lock{g_mutex};
  load();
  const auto it = g_shaders.find(key);
  if (it == g_shaders.end()) {
    return false;
  }
  *code = it->second;
  return true;
}

void store(rhi::ShaderStage stage, const std::string& source, const char* entry, const rhi::ShaderCode& code) {
  const Key key = key_of(stage, source, entry);
  std::unique_lock lock{g_mutex};
  load();
  if (!g_shaders.emplace(key, code).second || g_file == nullptr) {
    return;
  }
  const auto size = static_cast<uint32_t>(code.size());
  std::fwrite(&key, sizeof(key), 1, g_file);
  std::fwrite(&size, 4, 1, g_file);
  std::fwrite(code.data(), 1, size, g_file);
  std::fflush(g_file);
}
} // namespace aurora::gpu::shader_cache
