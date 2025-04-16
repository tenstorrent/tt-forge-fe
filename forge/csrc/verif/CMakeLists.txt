add_library(verif STATIC python_bindings.cpp verif_ops.cpp)

target_link_libraries(verif PUBLIC coverage_config)

target_compile_options(verif PRIVATE ${STATIC_LIB_FLAGS} ${TTFORGE_CSRC_CFLAGS})
