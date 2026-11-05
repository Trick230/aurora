add_library(aurora_core STATIC
        lib/aurora.cpp
        lib/imgui.cpp
        lib/logging.cpp
        lib/window.cpp
        lib/webgpu/gpu.cpp
)
add_library(aurora::core ALIAS aurora_core)
set_target_properties(aurora_core PROPERTIES FOLDER "aurora")

target_compile_definitions(aurora_core PUBLIC AURORA TARGET_PC)
target_include_directories(aurora_core PUBLIC include)
target_link_libraries(aurora_core PUBLIC fmt::fmt ${AURORA_SDL3_TARGET} xxhash imgui aurora_rhi)
target_link_libraries(aurora_core PRIVATE absl::btree absl::flat_hash_map sqlite3)

