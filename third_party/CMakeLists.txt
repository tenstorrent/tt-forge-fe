### Build tt-mlir ###

set(METAL_LIB_DIR "${CMAKE_SOURCE_DIR}/third_party/tt-mlir/third_party/tt-metal/src/tt-metal-build/lib")

add_custom_target(build_tt_mlir ALL
    COMMAND ${CMAKE_COMMAND} -E env
    BUILD_TYPE=${CMAKE_BUILD_TYPE} # Export build type to use
    CXX_COMPILER=${CMAKE_CXX_COMPILER} # Export CXX compiler to use
    C_COMPILER=${CMAKE_C_COMPILER} # Export C compiler to use
    TTMLIR_TOOLCHAIN_DIR=${TTMLIR_TOOLCHAIN_DIR} # Export TTMLIR_TOOLCHAIN_DIR to use
    TTMLIR_VENV_DIR=${TTMLIR_VENV_DIR} # Export TTMLIR_VENV_DIR to use
    TTMLIR_ENABLE_RUNTIME=ON # Always build runtime
    bash ${CMAKE_CURRENT_SOURCE_DIR}/build_mlir.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tt-mlir
    BYPRODUCTS ${METAL_LIB_DIR}/_ttnn.so # Workaround how Ninja handles dependencies
    USES_TERMINAL
)

### end build tt-mlir ###

### Build TVM ###

add_custom_target(build_tvm ALL
    COMMAND bash -c ${CMAKE_CURRENT_SOURCE_DIR}/tvm/install.sh
    COMMENT "Installing TVM"
    USES_TERMINAL
)

### end build TVM ###
