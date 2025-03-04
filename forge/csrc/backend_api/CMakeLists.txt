add_library(backend_api
    STATIC
    arch_type.cpp
    backend_api.cpp)

target_link_libraries(backend_api shared_utils)
target_link_libraries(backend_api coverage_config)
add_dependencies(backend_api shared_utils)
target_compile_options(backend_api PRIVATE ${STATIC_LIB_FLAGS} ${TTFORGE_CSRC_CFLAGS})
