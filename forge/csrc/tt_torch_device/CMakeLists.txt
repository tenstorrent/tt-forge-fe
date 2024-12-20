add_library(tt_torch_device
    STATIC
    tt_device.cpp
    torch_device_impl.cpp
    python_bindings.cpp)

target_compile_options(tt_torch_device PRIVATE ${STATIC_LIB_FLAGS} ${TTFORGE_CSRC_CFLAGS})
add_dependencies(tt_torch_device tt-mlir)
