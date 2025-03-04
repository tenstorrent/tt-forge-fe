add_library(graph
    STATIC
    defines.cpp
    edge.cpp
    graph.cpp
    node.cpp
    node_types.cpp
    shape.cpp
    utils.cpp
    python_bindings.cpp)

target_link_libraries(graph PUBLIC coverage_config)

target_compile_options(graph PRIVATE ${STATIC_LIB_FLAGS} ${TTFORGE_CSRC_CFLAGS})
