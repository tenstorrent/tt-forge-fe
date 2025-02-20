add_library(runtime STATIC runtime.cpp tt_device.cpp python_bindings.cpp)
add_dependencies(runtime tt-mlir)

target_compile_options(runtime PRIVATE ${STATIC_LIB_FLAGS} ${TTFORGE_CSRC_CFLAGS})
