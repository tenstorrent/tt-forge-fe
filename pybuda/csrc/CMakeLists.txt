set(PYTHON_SITE_PACKAGES_DIR ${PYBUDA_VENV_DIR}/lib/${PYBUDA_PYTHON_VERSION}/site-packages)

set(CMAKE_PREFIX_PATH 
    ${PYTHON_SITE_PACKAGES_DIR}/torch
    ${CMAKE_PREFIX_PATH})

find_package(Python3 COMPONENTS Development REQUIRED)
find_package(Torch REQUIRED)

set(TT_MLIR_ROOT_DIR ${CMAKE_SOURCE_DIR}/third_party/tt-mlir)
set(TTMLIR_INCLUDE_DIRS
    ${TT_MLIR_ROOT_DIR}/include
    ${TT_MLIR_ROOT_DIR}/build/include
    ${TT_MLIR_ROOT_DIR}/runtime/include)

set(PYBUDA_CSRC_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/third_party/fmt
    ${CMAKE_SOURCE_DIR}/third_party/pybind11/include
    ${CMAKE_SOURCE_DIR}/third_party/json
    ${CMAKE_SOURCE_DIR}/third_party/tt-mlir/build/include
    ${CMAKE_SOURCE_DIR}/third_party/tt-mlir/runtime/include
    ${CMAKE_SOURCE_DIR}/third_party/tt-mlir/include
    ${TTMLIR_TOOLCHAIN_DIR}/include
    ${Python3_INCLUDE_DIRS}
    ${TTMLIR_INCLUDE_DIRS}
)

include_directories(${PYBUDA_CSRC_INCLUDES})
# This is workaround for utils/assert.hpp using ##__VA_ARGS__ which is not supported by clang
include_directories(SYSTEM ${CMAKE_SOURCE_DIR})
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})

add_subdirectory(graph_lib)
add_subdirectory(autograd)
add_subdirectory(shared_utils)
add_subdirectory(backend_api)
add_subdirectory(reportify)
add_subdirectory(tt_torch_device)

### pybuda_csrc_objs ###

file(GLOB CPP_FILES 
    "pybuda_bindings.cpp"
    "buda_passes.cpp"
    "passes/*.cpp"
    "lower_to_buda/common.cpp"
)

add_library(pybuda_csrc_objs OBJECT ${CPP_FILES})
target_compile_options(pybuda_csrc_objs PRIVATE ${STATIC_LIB_FLAGS} ${PYBUDA_CSRC_CFLAGS})
add_dependencies(pybuda_csrc_objs build_tt_mlir)

### End of pybuda_csrc_objs ### 

######## pybuda_csrc ########

set(TTMLIR_LIB_DIR "${CMAKE_SOURCE_DIR}/third_party/tt-mlir/build/lib")
set(TTRUNTIME_LIB_DIR "${CMAKE_SOURCE_DIR}/third_party/tt-mlir/build/runtime/lib")
set(METAL_LIB_DIR "${CMAKE_SOURCE_DIR}/third_party/tt-mlir/third_party/tt-metal/src/tt-metal-build/lib")
set(TORCH_LIB_DIR "${PYBUDA_VENV_DIR}/lib/${PYBUDA_PYTHON_VERSION}/site-packages/torch/lib")

add_library(pybuda_csrc SHARED)

set(METAL_LIB_DIR "${CMAKE_SOURCE_DIR}/third_party/tt-mlir/third_party/tt-metal/src/tt-metal-build/lib")

# Because _ttnn library doesn't have lib prefix, this is workaround to make linking work
add_library(ttnn SHARED IMPORTED)
set_property(TARGET ttnn PROPERTY IMPORTED_LOCATION "${METAL_LIB_DIR}/_ttnn.so")

target_link_libraries(pybuda_csrc PRIVATE
    graph
    autograd
    shared_utils
    backend_api
    reportify
    tt_torch_device
    pybuda_csrc_objs
    LLVM
    MLIR
    MLIRTTDialect
    MLIRTTIRDialect
    MLIRTTNNDialect
    MLIRTTIRTransforms
    MLIRTTNNTransforms
    MLIRTTKernelDialect
    MLIRTTMetalDialect
    MLIRTTIRAnalysis
    TTRuntime
    TTRuntimeTTNN
    tt_metal
    device
    tt_eager
    ttnn
    tt_metal
    device
    tt_eager
    flatbuffers
    xml2
    curses
    z
    m
    torch_python
    c10
    ${PYBUDA_PYTHON_VERSION}
    ${TORCH_LIBRARIES}
)

target_compile_options(pybuda_csrc PRIVATE
    ${PYBUDA_CSRC_CFLAGS}
    ${CXXFLAGS}
    ${SHARED_LIB_FLAGS}
)

target_link_directories(pybuda_csrc PRIVATE
    ${TTMLIR_TOOLCHAIN_DIR}/lib
    ${TTMLIR_LIB_DIR}
    ${TTRUNTIME_LIB_DIR}
    ${METAL_LIB_DIR}
    ${TORCH_LIB_DIR})

### End of pybuda_csrc ###

#### Copy python module extension to pybuda directory ####
add_custom_target(run_after_pybuda_csrc ALL
    COMMAND mkdir -p ${PYBUDA_VENV_DIR}/lib/${PYBUDA_PYTHON_VERSION}/site-packages/pybuda
    COMMAND cp $<TARGET_FILE:pybuda_csrc> ${PYBUDA_VENV_DIR}/lib/${PYBUDA_PYTHON_VERSION}/site-packages/pybuda/_C.so
    COMMAND touch -r $<TARGET_FILE:pybuda_csrc> ${PYBUDA_VENV_DIR}/lib/${PYBUDA_PYTHON_VERSION}/site-packages/pybuda/_C.so
    COMMAND ln -sf ${PYBUDA_VENV_DIR}/lib/${PYBUDA_PYTHON_VERSION}/site-packages/pybuda/_C.so ${CMAKE_SOURCE_DIR}/pybuda/pybuda/_C.so
    COMMENT "Running run_after_pybuda_csrc to copy the python module extension to pybuda directory"
    USES_TERMINAL
)

add_dependencies(run_after_pybuda_csrc pybuda_csrc)
