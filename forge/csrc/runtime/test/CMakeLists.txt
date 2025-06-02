add_library(runtime-test STATIC runtime-test.cpp python_bindings.cpp)
add_dependencies(runtime-test runtime)

target_link_libraries(runtime-test PUBLIC coverage_config)

target_compile_options(runtime-test PRIVATE ${STATIC_LIB_FLAGS} ${TTFORGE_CSRC_CFLAGS})
