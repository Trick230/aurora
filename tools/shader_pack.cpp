// Compiles, ahead of time, the shaders of a game's pipeline seed and aurora's own
// pipelines for every backend:
// the shaders_<backend>.bin files aurora loads from its resources directory.
// Usage: aurora_shader_pack <initial_pipeline_cache.db> <out dir>
#include "../lib/gfx/ambient_occlusion.hpp"
#include "../lib/gfx/clear.hpp"
#include "../lib/gfx/common.hpp"
#include "../lib/gfx/tex_copy_conv.hpp"
#include "../lib/gfx/tex_palette_conv.hpp"
#include "../lib/gpu/gpu.hpp"
#include "../lib/gpu/shader_cache.hpp"
#include "../lib/gx/pipeline.hpp"
#include "../lib/gx/shader_hlsl.hpp"
#include "../lib/imgui.hpp"

#include <absl/container/flat_hash_set.h>
#include <sqlite3.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace aurora;
namespace shader_cache = aurora::gpu::shader_cache;

namespace {
bool read_configs(const char* path, std::vector<gx::PipelineConfig>* configs) {
  sqlite3* db = nullptr;
  if (sqlite3_open_v2(path, &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
    std::fprintf(stderr, "can't open %s: %s\n", path, sqlite3_errmsg(db));
    sqlite3_close(db);
    return false;
  }
  sqlite3_stmt* stmt = nullptr;
  bool ok = sqlite3_prepare_v2(db, "SELECT config FROM pipeline_cache WHERE type = ? AND config_version = ?", -1,
                               &stmt, nullptr) == SQLITE_OK;
  if (ok) {
    sqlite3_bind_int(stmt, 1, static_cast<int>(gfx::ShaderType::GX));
    sqlite3_bind_int(stmt, 2, static_cast<int>(gx::GXPipelineConfigVersion));
    int ret;
    while ((ret = sqlite3_step(stmt)) == SQLITE_ROW) {
      if (sqlite3_column_bytes(stmt, 0) != static_cast<int>(sizeof(gx::PipelineConfig))) {
        continue;
      }
      gx::PipelineConfig config;
      std::memcpy(&config, sqlite3_column_blob(stmt, 0), sizeof(config));
      configs->push_back(config);
    }
    ok = ret == SQLITE_DONE;
  }
  if (!ok) {
    std::fprintf(stderr, "can't read %s: %s\n", path, sqlite3_errmsg(db));
  }
  sqlite3_finalize(stmt);
  sqlite3_close(db);
  return ok;
}

struct Job {
  rhi::ShaderStage stage;
  const char* entry;
  const std::string* source;
};
} // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: aurora_shader_pack <initial_pipeline_cache.db> <out dir>\n");
    return 2;
  }
  const auto start = std::chrono::steady_clock::now();
  std::vector<gx::PipelineConfig> configs;
  if (!read_configs(argv[1], &configs)) {
    return 1;
  }
  // The sources the game builds for these pipelines, each once, and aurora's own.
  std::vector<std::string> sources;
  {
    absl::flat_hash_set<std::string> seen;
    const auto add = [&](std::string source) {
      if (seen.insert(source).second) {
        sources.push_back(std::move(source));
      }
    };
    for (const auto& config : configs) {
      add(gx::hlsl::build_shader_source(config.shaderConfig, gx::dst_alpha_mode(config)));
    }
    for (const auto mode : {gx::DstAlphaMode::None, gx::DstAlphaMode::Replace, gx::DstAlphaMode::DualSource}) {
      add(gx::hlsl::uber_source(mode));
    }
    for (auto* const list : {gfx::ao::shader_sources, gfx::clear::shader_sources, gfx::tex_copy_conv::shader_sources,
                             gfx::tex_palette_conv::shader_sources, gpu::shader_sources, imgui::shader_sources}) {
      for (auto& source : list()) {
        add(std::move(source));
      }
    }
  }
  std::vector<Job> jobs;
  for (const auto& source : sources) {
    jobs.push_back({rhi::ShaderStage::Vertex, "vs_main", &source});
    jobs.push_back({rhi::ShaderStage::Fragment, "fs_main", &source});
  }

  const std::filesystem::path outDir{argv[2]};
  std::error_code ec;
  std::filesystem::create_directories(outDir, ec);
  for (const auto backend : {rhi::Backend::D3D12, rhi::Backend::Vulkan}) {
    std::vector<rhi::ShaderCode> codes(jobs.size());
    std::atomic<size_t> next{0};
    std::atomic<bool> failed{false};
    std::mutex print;
    const auto worker = [&] {
      for (size_t i = next++; i < jobs.size() && !failed; i = next++) {
        std::string error;
        if (!rhi::compile_shader(backend, jobs[i].stage, *jobs[i].source, jobs[i].entry, &codes[i], &error)) {
          std::lock_guard lock{print};
          std::fprintf(stderr, "%s doesn't compile: %s\n%s\n", jobs[i].entry, error.c_str(), jobs[i].source->c_str());
          failed = true;
        }
      }
    };
    std::vector<std::thread> threads(std::max(1u, std::thread::hardware_concurrency()));
    for (auto& thread : threads) {
      thread = std::thread{worker};
    }
    for (auto& thread : threads) {
      thread.join();
    }
    if (failed) {
      return 1;
    }
    shader_cache::Records records;
    records.reserve(jobs.size());
    for (size_t i = 0; i < jobs.size(); ++i) {
      records.emplace_back(shader_cache::key(jobs[i].stage, *jobs[i].source, jobs[i].entry), std::move(codes[i]));
    }
    const auto path = outDir / shader_cache::file_name(backend);
    if (!shader_cache::write_file(path, shader_cache::salt(backend), records)) {
      std::fprintf(stderr, "can't write %s\n", path.string().c_str());
      return 1;
    }
  }
  const auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
  std::printf("aurora_shader_pack: %zu GX pipelines, %zu sources, %zu shaders per backend in %lld ms\n", configs.size(),
              sources.size(), jobs.size(), static_cast<long long>(ms));
  return 0;
}
