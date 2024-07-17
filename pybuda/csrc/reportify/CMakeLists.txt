add_library(reportify
    STATIC
    reportify.cpp
    paths.cpp
    to_json.cpp)

target_compile_options(reportify PRIVATE ${STATIC_LIB_FLAGS} ${PYBUDA_CSRC_CFLAGS})

