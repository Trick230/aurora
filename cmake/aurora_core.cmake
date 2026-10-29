add_library(aurora_core STATIC
        lib/aurora.cpp
        lib/imgui.cpp
        lib/logging.cpp
        lib/window.cpp
        lib/webgpu/gpu.cpp
        lib/webgpu/gpu_cache.cpp
        lib/dawn/BackendBinding.cpp
)
add_library(aurora::core ALIAS aurora_core)
set_target_properties(aurora_core PROPERTIES FOLDER "aurora")

target_compile_definitions(aurora_core PUBLIC AURORA TARGET_PC WEBGPU_DAWN)
target_include_directories(aurora_core PUBLIC include)
target_link_libraries(aurora_core PUBLIC fmt::fmt ${AURORA_SDL3_TARGET} xxhash imgui)
target_link_libraries(aurora_core PRIVATE absl::btree absl::flat_hash_map sqlite3 dawn::webgpu_dawn)

if (DAWN_ENABLE_VULKAN)
    target_compile_definitions(aurora_core PRIVATE DAWN_ENABLE_BACKEND_VULKAN)
endif ()
if (DAWN_ENABLE_METAL)
    target_compile_definitions(aurora_core PRIVATE DAWN_ENABLE_BACKEND_METAL)
    target_sources(aurora_core PRIVATE lib/dawn/MetalBinding.mm)
    set_source_files_properties(lib/dawn/MetalBinding.mm PROPERTIES COMPILE_FLAGS -fobjc-arc)
    target_link_options(aurora_core PUBLIC "LINKER:-weak_framework,Metal")
endif ()
if (DAWN_ENABLE_D3D12)
    target_compile_definitions(aurora_core PRIVATE DAWN_ENABLE_BACKEND_D3D12)
endif ()
