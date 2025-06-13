add_library(runtime_testutils STATIC runtime_testutils.cpp python_bindings.cpp)
add_dependencies(runtime_testutils runtime)

target_link_libraries(runtime_testutils PUBLIC coverage_config)

target_compile_options(runtime_testutils PRIVATE ${STATIC_LIB_FLAGS} ${TTFORGE_CSRC_CFLAGS})
