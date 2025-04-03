add_library(runtime-testutils STATIC runtime-testutils.cpp python_bindings.cpp)
add_dependencies(runtime-testutils runtime)

target_link_libraries(runtime-testutils PUBLIC coverage_config)

target_compile_options(runtime-testutils PRIVATE ${STATIC_LIB_FLAGS} ${TTFORGE_CSRC_CFLAGS})
