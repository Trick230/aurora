add_library(aurora_gx STATIC
        lib/gfx/ambient_occlusion.cpp
        lib/gfx/clear.cpp
        lib/gfx/common.cpp
        lib/gfx/frame_interp.cpp
        lib/gfx/pipeline_cache.cpp
        lib/gfx/render_worker.cpp
        lib/gfx/tex_copy_conv.cpp
        lib/gfx/tex_palette_conv.cpp
        lib/gfx/texture.cpp
        lib/gfx/texture_format.cpp
        lib/gfx/texture_convert.cpp
        lib/gfx/dds_io.cpp
        lib/gfx/png_io.cpp
        lib/gfx/texture_replacement.cpp
        lib/gx/attr_fmt.cpp
        lib/gx/command_processor.cpp
        lib/gx/efb_copy.cpp
        lib/gx/fifo.cpp
        lib/gx/gx.cpp
        lib/gx/pipeline.cpp
        lib/gx/recomp.cpp
        lib/gx/shader_hlsl.cpp
        lib/gx/shader_info.cpp
        lib/gx/ubershader.cpp
        lib/gx/ubershader_hlsl.cpp
)
add_library(aurora::gx ALIAS aurora_gx)
set_target_properties(aurora_gx PROPERTIES FOLDER "aurora")

target_link_libraries(aurora_gx PUBLIC aurora::core xxhash)
target_link_libraries(aurora_gx PRIVATE absl::btree absl::flat_hash_map sqlite3 PNG::PNG)
