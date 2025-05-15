add_library(shared_utils
    STATIC
    placement_printer.cpp
    pretty_table.cpp
    sparse_matmul_utils.cpp
    string_extension.cpp)

target_link_libraries(shared_utils PUBLIC coverage_config)

target_compile_options(shared_utils PRIVATE ${STATIC_LIB_FLAGS} ${TTFORGE_CSRC_CFLAGS})
