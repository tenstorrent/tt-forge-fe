### Build tt-mlir ###
include(ExternalProject)

set(METAL_LIB_DIR "${CMAKE_SOURCE_DIR}/third_party/tt-mlir/third_party/tt-metal/src/tt-metal-build/lib")

set(TTMLIR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tt-mlir")
set(TTMLIR_INSTALL_PREFIX "${TTMLIR_SOURCE_DIR}/install")

ExternalProject_Add(
    tt-mlir
    SOURCE_DIR ${TTMLIR_SOURCE_DIR}
    BINARY_DIR ${TTMLIR_SOURCE_DIR}/build
    INSTALL_DIR ${TTMLIR_INSTALL_PREFIX}
    PATCH_COMMAND bash -c "source ${TTMLIR_SOURCE_DIR}/env/activate"
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_INSTALL_PREFIX=${TTMLIR_INSTALL_PREFIX}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DTTMLIR_TOOLCHAIN_DIR=${TTMLIR_TOOLCHAIN_DIR}
        -DCMAKE_MODULE_PATH=${TTMLIR_TOOLCHAIN_DIR}/lib/cmake
        -DTTMLIR_ENABLE_RUNTIME=ON
        -DTT_RUNTIME_DEBUG=${TTMLIR_RUNTIME_DEBUG}
    LOG_CONFIGURE ON
    LOG_BUILD ON
    LOG_INSTALL ON
)

ExternalProject_Add_StepTargets(tt-mlir configure)
set_target_properties(tt-mlir-configure PROPERTIES EXCLUDE_FROM_ALL TRUE)

set(TTMLIR_OUTPUT_LIB_DIR ${CMAKE_INSTALL_PREFIX}/lib/TTMLIR)
message(STATUS "TTMLIR_OUTPUT_LIB_DIR: ${TTMLIR_OUTPUT_LIB_DIR}")
message(STATUS "TTMLIR_INSTALL_PREFIX: ${TTMLIR_INSTALL_PREFIX}")
install(DIRECTORY ${TTMLIR_INSTALL_PREFIX} DESTINATION ${TTMLIR_OUTPUT_LIB_DIR})

# install(
#     FILES
#         ${TTMLIR_TOOLCHAIN_DIR}/lib/libLLVM.so
#         ${TTMLIR_TOOLCHAIN_DIR}/lib/libMLIR.so
#     DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
# )

install(CODE
  "
    file(INSTALL
      DESTINATION \"${CMAKE_INSTALL_PREFIX}/lib\"
      TYPE SHARED_LIBRARY
      FILES
          \"${TTMLIR_TOOLCHAIN_DIR}/lib/libLLVM.so\"
          \"${TTMLIR_TOOLCHAIN_DIR}/lib/libMLIR.so\"
      FOLLOW_SYMLINK_CHAIN
    )
  "
)
#
# set_target_properties(tt-mlir PROPERTIES EXCLUDE_FROM_ALL TRUE)
#
# list(APPEND CMAKE_MODULE_PATH ${TTMLIR_INSTALL_PREFIX})
# find_package(TTMLIR CONFIG REQUIRED)
# ExternalProject_Add_StepTargets(tt-mlir install)

# add_library(TTMLIR SHARED IMPORTED GLOBAL DEPENDS tt-mlir-configure)
# set_target_properties(TTMLIR PROPERTIES IMPORTED_LOCATION ${TTMLIR_OUTPUT_LIB_DIR}/libTTMLIR.so)

### end build tt-mlir ###

### Build TVM ###

add_custom_target(build_tvm ALL
    COMMAND bash -c ${CMAKE_CURRENT_SOURCE_DIR}/tvm/install.sh
    COMMENT "Installing TVM"
    USES_TERMINAL
)

### end build TVM ###
