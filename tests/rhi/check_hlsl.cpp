// Compiles every .hlsl file of a folder, vs_main and fs_main, for every backend, and
// says what doesn't compile. Usage: check_hlsl <dir>
#include "rhi.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace aurora::rhi;

int main(int argc, char** argv) {
  if (argc != 2) {
    std::printf("usage: check_hlsl <dir>\n");
    return 2;
  }
  std::vector<std::filesystem::path> files;
  for (const auto& entry : std::filesystem::directory_iterator(argv[1])) {
    if (entry.path().extension() == ".hlsl")
      files.push_back(entry.path());
  }
  if (files.empty()) {
    std::printf("no .hlsl files in %s\n", argv[1]);
    return 2;
  }
  std::atomic<size_t> next{0};
  std::atomic<int> failed{0}, shown{0};
  std::atomic<uint64_t> bytes[2]{};
  std::mutex print;
  const auto worker = [&] {
    for (size_t i = next++; i < files.size(); i = next++) {
      std::ifstream stream(files[i], std::ios::binary);
      std::stringstream text;
      text << stream.rdbuf();
      const std::string source = text.str();
      bool ok = true;
      for (const Backend backend : {Backend::D3D12, Backend::Vulkan}) {
        for (const auto& [stage, entry] : {std::pair{ShaderStage::Vertex, "vs_main"}, std::pair{ShaderStage::Fragment, "fs_main"}}) {
          ShaderCode code;
          std::string error;
          if (compile_shader(backend, stage, source, entry, &code, &error)) {
            bytes[backend == Backend::D3D12 ? 0 : 1] += code.size();
          } else {
            ok = false;
            if (shown++ < 6) {
              std::lock_guard lock{print};
              std::printf("%s %s %s:\n%.1200s\n", files[i].filename().string().c_str(), entry,
                          backend == Backend::D3D12 ? "D3D12" : "Vulkan", error.c_str());
            }
          }
        }
      }
      if (!ok)
        ++failed;
    }
  };
  const auto start = std::chrono::steady_clock::now();
  std::vector<std::thread> threads;
  for (int i = 0; i < 4; ++i)
    threads.emplace_back(worker);
  for (auto& thread : threads)
    thread.join();
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::printf("%zu sources, %d fail; %.1f s; DXIL %.1f MB, SPIR-V %.1f MB\n", files.size(), failed.load(), seconds,
              bytes[0].load() / 1e6, bytes[1].load() / 1e6);
  return failed ? 1 : 0;
}
