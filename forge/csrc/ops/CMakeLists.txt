add_library(ops
    STATIC
    op.cpp
    op_abs.cpp
    op_constant.cpp
    op_matmul.cpp
    python_bindings.cpp)

target_link_libraries(ops PUBLIC coverage_config)

target_compile_options(ops PRIVATE ${STATIC_LIB_FLAGS} ${TTFORGE_CSRC_CFLAGS})
