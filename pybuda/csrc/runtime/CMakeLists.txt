add_library(runtime STATIC runtime.cpp tt_device.cpp python_bindings.cpp)
add_dependencies(runtime build_tt_mlir)

target_compile_options(runtime PRIVATE ${STATIC_LIB_FLAGS} ${PYBUDA_CSRC_CFLAGS})
