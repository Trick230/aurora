include_guard(GLOBAL)

# aurora_copy_runtime_dlls(<executable_target> [<additional_targets>...])
#
# POST_BUILD command that copies all TARGET_RUNTIME_DLLS from the given targets
# next to the executable, plus the DLLs the RHI loads at run time
# (dxcompiler.dll, dxil.dll).
function(aurora_copy_runtime_dlls target)
  if (NOT WIN32)
    return()
  endif ()

  # Copy TARGET_RUNTIME_DLLS for each target
  foreach (_t IN ITEMS ${target} ${ARGN})
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${_t}> $<TARGET_FILE_DIR:${target}>
        COMMAND_EXPAND_LISTS
    )
  endforeach ()

  # The DXC DLLs the RHI loads at runtime
  get_property(_rhi_files GLOBAL PROPERTY AURORA_RHI_RUNTIME_FILES)
  foreach (_file IN LISTS _rhi_files)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_file}" $<TARGET_FILE_DIR:${target}>
    )
  endforeach ()
endfunction()

# aurora_install_runtime_dlls(<executable_target> <install_destination>)
#
# INSTALL rule that copies all DLLs already placed next to the built target
# into the install destination.
function(aurora_install_runtime_dlls target install_destination)
  if (NOT WIN32)
    return()
  endif ()

  install(CODE
    "file(GLOB _aurora_runtime_dlls \"$<TARGET_FILE_DIR:${target}>/*.dll\")\n\
if(_aurora_runtime_dlls)\n\
  file(INSTALL DESTINATION \"${install_destination}\" TYPE FILE FILES \${_aurora_runtime_dlls})\n\
endif()"
  )
endfunction()
