add_library(reportify
    STATIC
    reportify.cpp
    paths.cpp
    to_json.cpp)

target_link_libraries(reportify PUBLIC coverage_config)

target_compile_options(reportify PRIVATE ${STATIC_LIB_FLAGS} ${TTFORGE_CSRC_CFLAGS})
