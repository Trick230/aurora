#include "rhi.hpp"

#include <windows.h>

#include <objidl.h>

#include <dxcapi.h>
#include <wrl/client.h>

#include <string>
#include <vector>

namespace aurora::rhi {
namespace {

using Microsoft::WRL::ComPtr;

struct Compiler {
  ComPtr<IDxcUtils> utils;
  ComPtr<IDxcCompiler3> compiler;
};

// Generated shaders get compiled on several threads, and a DXC instance serves
// one at a time.
Compiler* thread_compiler(std::string* error) {
  static const auto create = reinterpret_cast<DxcCreateInstanceProc>(
      reinterpret_cast<void*>(GetProcAddress(LoadLibraryW(L"dxcompiler.dll"), "DxcCreateInstance")));
  if (!create) {
    *error = "dxcompiler.dll not found";
    return nullptr;
  }
  thread_local Compiler instance;
  if (!instance.compiler && (FAILED(create(CLSID_DxcUtils, IID_PPV_ARGS(&instance.utils))) ||
                             FAILED(create(CLSID_DxcCompiler, IID_PPV_ARGS(&instance.compiler))))) {
    *error = "could not create a DXC instance";
    return nullptr;
  }
  return &instance;
}

} // namespace

std::string shader_compiler_version() {
  // Bumped whenever compile_shader's arguments change.
  std::string version = "args1";
  std::string error;
  Compiler* dxc = thread_compiler(&error);
  if (!dxc)
    return version;
  ComPtr<IDxcVersionInfo> info;
  if (SUCCEEDED(dxc->compiler.As(&info))) {
    UINT32 major = 0, minor = 0;
    info->GetVersion(&major, &minor);
    version += " dxc " + std::to_string(major) + "." + std::to_string(minor);
  }
  ComPtr<IDxcVersionInfo2> commit;
  if (SUCCEEDED(dxc->compiler.As(&commit))) {
    UINT32 count = 0;
    char* hash = nullptr;
    if (SUCCEEDED(commit->GetCommitInfo(&count, &hash)) && hash) {
      version += " " + std::to_string(count) + " " + hash;
      CoTaskMemFree(hash);
    }
  }
  return version;
}

bool compile_shader(Backend backend, ShaderStage stage, const std::string& source, const std::string& entry,
                    ShaderCode* code, std::string* error) {
  Compiler* dxc = thread_compiler(error);
  if (!dxc)
    return false;
  const std::wstring wide_entry(entry.begin(), entry.end());
  std::vector<LPCWSTR> arguments = {L"-E", wide_entry.c_str(), L"-T", stage == ShaderStage::Vertex ? L"vs_6_0" : L"ps_6_0",
                                    L"-O3"};
  if (backend == Backend::Vulkan) {
    const LPCWSTR vulkan[] = {L"-spirv",      L"-fspv-target-env=vulkan1.3",
                              L"-fvk-use-dx-layout",
                              L"-fvk-t-shift", L"1000", L"all",
                              L"-fvk-s-shift", L"2000", L"all",
                              L"-fvk-u-shift", L"3000", L"all"};
    arguments.insert(arguments.end(), std::begin(vulkan), std::end(vulkan));
  }
  DxcBuffer buffer{source.data(), source.size(), DXC_CP_UTF8};
  ComPtr<IDxcResult> result;
  if (FAILED(dxc->compiler->Compile(&buffer, arguments.data(), static_cast<UINT32>(arguments.size()), nullptr,
                                    IID_PPV_ARGS(&result)))) {
    *error = "DXC failed to run";
    return false;
  }
  HRESULT status = E_FAIL;
  result->GetStatus(&status);
  if (FAILED(status)) {
    ComPtr<IDxcBlobUtf8> messages;
    result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&messages), nullptr);
    *error = messages && messages->GetStringLength() ? messages->GetStringPointer() : "DXC failed";
    return false;
  }
  ComPtr<IDxcBlob> object;
  if (FAILED(result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&object), nullptr)) || !object) {
    *error = "DXC gave no code";
    return false;
  }
  const auto* bytes = static_cast<const uint8_t*>(object->GetBufferPointer());
  code->assign(bytes, bytes + object->GetBufferSize());
  return true;
}

} // namespace aurora::rhi
