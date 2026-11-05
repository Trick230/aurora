include_guard(GLOBAL)
include(FetchContent)

# The native renderer's layer over D3D12 and Vulkan. Only headers and single-file
# libraries come from these projects, so none of their own targets are configured.
set(_aurora_rhi_vulkan_tag "vulkan-sdk-1.4.363.0")
FetchContent_Declare(aurora_vulkan_headers
        GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
        GIT_TAG ${_aurora_rhi_vulkan_tag} GIT_SHALLOW TRUE SOURCE_SUBDIR _none)
FetchContent_Declare(aurora_volk
        GIT_REPOSITORY https://github.com/zeux/volk.git
        GIT_TAG ${_aurora_rhi_vulkan_tag} GIT_SHALLOW TRUE SOURCE_SUBDIR _none)
FetchContent_Declare(aurora_vma
        GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
        GIT_TAG v3.5.0 GIT_SHALLOW TRUE SOURCE_SUBDIR _none)
FetchContent_Declare(aurora_dxc
        URL https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.9.2609/dxc_2026_09_29.zip
        URL_HASH SHA256=ad31b1fc8443175d204f77a611fdb3ef2ec42759bdc2f1167368de24a4a7e7f1
        DOWNLOAD_EXTRACT_TIMESTAMP FALSE SOURCE_SUBDIR _none)
FetchContent_Declare(aurora_d3d12ma
        GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/D3D12MemoryAllocator.git
        GIT_TAG v3.2.1 GIT_SHALLOW TRUE SOURCE_SUBDIR _none)
FetchContent_MakeAvailable(aurora_vulkan_headers aurora_volk aurora_vma aurora_dxc)

# DXC runs at run time, for the shaders generated then; these go next to the executable.
set(AURORA_RHI_RUNTIME_FILES "${aurora_dxc_SOURCE_DIR}/bin/x64/dxcompiler.dll" "${aurora_dxc_SOURCE_DIR}/bin/x64/dxil.dll")

set(_aurora_rhi_dir "${CMAKE_CURRENT_LIST_DIR}/../lib/rhi")
add_library(aurora_rhi STATIC
        ${_aurora_rhi_dir}/rhi.cpp
        ${_aurora_rhi_dir}/vulkan.cpp
        ${_aurora_rhi_dir}/shader_compiler.cpp
        ${aurora_volk_SOURCE_DIR}/volk.c)
set_target_properties(aurora_rhi PROPERTIES FOLDER "aurora")
target_include_directories(aurora_rhi PUBLIC ${_aurora_rhi_dir})
target_include_directories(aurora_rhi SYSTEM PRIVATE
        ${aurora_vulkan_headers_SOURCE_DIR}/include ${aurora_volk_SOURCE_DIR} ${aurora_vma_SOURCE_DIR}/include)
if (WIN32)
    FetchContent_MakeAvailable(aurora_d3d12ma)
    target_sources(aurora_rhi PRIVATE ${_aurora_rhi_dir}/d3d12.cpp ${aurora_d3d12ma_SOURCE_DIR}/src/D3D12MemAlloc.cpp)
    target_include_directories(aurora_rhi SYSTEM PRIVATE ${aurora_d3d12ma_SOURCE_DIR}/include)
    target_compile_definitions(aurora_rhi PRIVATE VK_USE_PLATFORM_WIN32_KHR NOMINMAX WIN32_LEAN_AND_MEAN)
    target_link_libraries(aurora_rhi PRIVATE d3d12 dxgi dxguid)
else ()
    target_link_libraries(aurora_rhi PRIVATE ${CMAKE_DL_LIBS})
endif ()
