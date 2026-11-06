#include "shader_cache.hpp"

#include "gpu.hpp"
#include "../internal.hpp"

#include <absl/container/flat_hash_map.h>
#include <xxhash.h>

#include <cstdio>
#include <cstring>
#include <mutex>
#include <shared_mutex>

namespace aurora::gpu::shader_cache {
namespace {
Module Log("aurora::gpu::shader_cache");

// A file: this magic, the salt's length and the salt, then records of a 128-bit
// key, a 32-bit size and the code. A record cut short ends the file.
constexpr char Magic[8] = {'A', 'U', 'R', 'S', 'H', 'D', 'R', '1'};

std::shared_mutex g_mutex;
absl::flat_hash_map<Key, rhi::ShaderCode> g_shaders;
std::FILE* g_file = nullptr; // this machine's cache, open to append
bool g_loaded = false;

void write_header(std::FILE* file, const std::string& salt) {
  const auto length = static_cast<uint32_t>(salt.size());
  std::fwrite(Magic, 1, sizeof(Magic), file);
  std::fwrite(&length, 4, 1, file);
  std::fwrite(salt.data(), 1, length, file);
}

void write_record(std::FILE* file, const Key& key, const rhi::ShaderCode& code) {
  const auto size = static_cast<uint32_t>(code.size());
  std::fwrite(&key, sizeof(key), 1, file);
  std::fwrite(&size, 4, 1, file);
  std::fwrite(code.data(), 1, size, file);
}

// Under the exclusive lock.
void load() {
  if (g_loaded) {
    return;
  }
  g_loaded = true;
  const rhi::Backend backend = g_device->backend();
  const std::string expected = salt(backend);
  Records shipped, local;
  if (g_config.resourcesPath != nullptr && g_config.resourcesPath[0] != '\0') {
    read_file(std::filesystem::path{g_config.resourcesPath} / file_name(backend), expected, &shipped);
  }
  for (auto& [key, code] : shipped) {
    g_shaders.insert_or_assign(key, std::move(code));
  }
  if (g_config.cachePath == nullptr || g_config.cachePath[0] == '\0') {
    Log.info("{} shaders shipped, no cache directory", shipped.size());
    return;
  }
  const auto path = std::filesystem::path{g_config.cachePath} / file_name(backend);
  if (read_file(path, expected, &local)) {
    g_file = std::fopen(path.string().c_str(), "ab");
  } else {
    // Missing, or from another compiler: started over.
    g_file = std::fopen(path.string().c_str(), "wb");
    if (g_file != nullptr) {
      write_header(g_file, expected);
      std::fflush(g_file);
    }
  }
  const size_t localCount = local.size();
  for (auto& [key, code] : local) {
    g_shaders.insert_or_assign(key, std::move(code));
  }
  Log.info("{} shaders shipped, {} compiled here before", shipped.size(), localCount);
}
} // namespace

Key key(rhi::ShaderStage stage, const std::string& source, const char* entry) {
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

const char* file_name(rhi::Backend backend) {
  return backend == rhi::Backend::Vulkan ? "shaders_vulkan.bin" : "shaders_d3d12.bin";
}

std::string salt(rhi::Backend backend) {
  return std::string{backend == rhi::Backend::Vulkan ? "vulkan" : "d3d12"} + "|" + rhi::shader_compiler_version();
}

bool read_file(const std::filesystem::path& path, const std::string& expected, Records* records) {
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
    records->emplace_back(key, std::move(code));
  }
  std::fclose(file);
  return ok;
}

bool write_file(const std::filesystem::path& path, const std::string& salt, const Records& records) {
  std::FILE* const file = std::fopen(path.string().c_str(), "wb");
  if (file == nullptr) {
    return false;
  }
  write_header(file, salt);
  for (const auto& [key, code] : records) {
    write_record(file, key, code);
  }
  return std::fclose(file) == 0;
}

bool find(rhi::ShaderStage stage, const std::string& source, const char* entry, rhi::ShaderCode* code) {
  const Key id = key(stage, source, entry);
  {
    std::shared_lock lock{g_mutex};
    if (g_loaded) {
      const auto it = g_shaders.find(id);
      if (it == g_shaders.end()) {
        return false;
      }
      *code = it->second;
      return true;
    }
  }
  std::unique_lock lock{g_mutex};
  load();
  const auto it = g_shaders.find(id);
  if (it == g_shaders.end()) {
    return false;
  }
  *code = it->second;
  return true;
}

void store(rhi::ShaderStage stage, const std::string& source, const char* entry, const rhi::ShaderCode& code) {
  const Key id = key(stage, source, entry);
  std::unique_lock lock{g_mutex};
  load();
  if (!g_shaders.emplace(id, code).second || g_file == nullptr) {
    return;
  }
  write_record(g_file, id, code);
  std::fflush(g_file);
}
} // namespace aurora::gpu::shader_cache
