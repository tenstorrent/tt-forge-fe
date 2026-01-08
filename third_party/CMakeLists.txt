### Build tt-mlir ###
include(ExternalProject)

option(TTMLIR_USE_EXTERNAL "Use an external pre-built tt-mlir instead of building from submodule" OFF)

if(DEFINED TTMLIR_SOURCE_DIR AND NOT TTMLIR_SOURCE_DIR STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/tt-mlir")
    set(TTMLIR_USE_EXTERNAL ON)
    message(STATUS "Using external tt-mlir from: ${TTMLIR_SOURCE_DIR}")
else()
    set(TTMLIR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tt-mlir")
endif()

set(TTMLIR_BUILD_DIR "${TTMLIR_SOURCE_DIR}/build")
set(TTMLIR_INSTALL_PREFIX "${TTMLIR_BUILD_DIR}/install")
set(METAL_LIB_DIR "${TTMLIR_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/build/lib")

if(TTMLIR_USE_EXTERNAL)
    if(NOT EXISTS "${TTMLIR_BUILD_DIR}")
        message(FATAL_ERROR
            "External tt-mlir build directory not found: ${TTMLIR_BUILD_DIR}\n"
            "Please ensure tt-mlir is built:\n"
            "  cd ${TTMLIR_SOURCE_DIR}\n"
            "  cmake --build build")
    endif()

    # Auto-install SharedLib component if needed
    if(NOT EXISTS "${TTMLIR_INSTALL_PREFIX}/lib/libTTMLIRCompiler.so")
        message(STATUS "Installing tt-mlir SharedLib component...")
        execute_process(
            COMMAND ${CMAKE_COMMAND} --install ${TTMLIR_BUILD_DIR} --prefix ${TTMLIR_INSTALL_PREFIX} --component SharedLib
            WORKING_DIRECTORY ${TTMLIR_SOURCE_DIR}
            RESULT_VARIABLE INSTALL_RESULT
        )
        if(NOT INSTALL_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to install tt-mlir SharedLib component")
        endif()
    endif()

    # Auto-install TTNNStandalone component if needed
    if(NOT EXISTS "${TTMLIR_INSTALL_PREFIX}/lib/libTTNNCompileSo.so")
        message(STATUS "Installing tt-mlir TTNNStandalone component...")
        execute_process(
            COMMAND ${CMAKE_COMMAND} --install ${TTMLIR_BUILD_DIR} --prefix ${TTMLIR_INSTALL_PREFIX} --component TTNNStandalone
            WORKING_DIRECTORY ${TTMLIR_SOURCE_DIR}
            RESULT_VARIABLE INSTALL_RESULT
        )
        if(NOT INSTALL_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to install tt-mlir TTNNStandalone component")
        endif()
    endif()

    # Copy TTMLIRRuntime if not in install dir (it's not part of standard install components)
    if(NOT EXISTS "${TTMLIR_INSTALL_PREFIX}/lib/libTTMLIRRuntime.so")
        if(EXISTS "${TTMLIR_BUILD_DIR}/runtime/lib/libTTMLIRRuntime.so")
            message(STATUS "Copying libTTMLIRRuntime.so to install directory...")
            file(COPY "${TTMLIR_BUILD_DIR}/runtime/lib/libTTMLIRRuntime.so"
                 DESTINATION "${TTMLIR_INSTALL_PREFIX}/lib/")
        else()
            message(FATAL_ERROR
                "libTTMLIRRuntime.so not found in ${TTMLIR_BUILD_DIR}/runtime/lib/\n"
                "Please ensure tt-mlir was built with TTMLIR_ENABLE_RUNTIME=ON")
        endif()
    endif()

    set(REQUIRED_LIBS
        "${TTMLIR_INSTALL_PREFIX}/lib/libTTMLIRCompiler.so"
        "${TTMLIR_INSTALL_PREFIX}/lib/libTTNNCompileSo.so"
        "${TTMLIR_INSTALL_PREFIX}/lib/libTTMLIRRuntime.so"
    )
    foreach(lib ${REQUIRED_LIBS})
        if(NOT EXISTS ${lib})
            message(FATAL_ERROR "Required library still missing after auto-install: ${lib}")
        endif()
    endforeach()

    add_custom_target(tt-mlir
        COMMENT "Using external tt-mlir from ${TTMLIR_SOURCE_DIR}"
    )

    message(STATUS "External tt-mlir validated successfully")
    message(STATUS "  Source: ${TTMLIR_SOURCE_DIR}")
    message(STATUS "  Build:  ${TTMLIR_BUILD_DIR}")
    message(STATUS "  Install: ${TTMLIR_INSTALL_PREFIX}")
else()
    # Build tt-mlir from submodule (default behavior)
    ExternalProject_Add(
        tt-mlir
        SOURCE_DIR ${TTMLIR_SOURCE_DIR}
        BINARY_DIR ${TTMLIR_BUILD_DIR}
        INSTALL_COMMAND ${CMAKE_COMMAND} --install ${TTMLIR_BUILD_DIR} --component SharedLib && ${CMAKE_COMMAND} --install ${TTMLIR_BUILD_DIR} --component TTNNStandalone
        CMAKE_GENERATOR Ninja
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
            -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
            -DCMAKE_INSTALL_PREFIX=${TTMLIR_INSTALL_PREFIX}
            -DTTMLIR_TOOLCHAIN_DIR=${TTMLIR_TOOLCHAIN_DIR}
            -DTTMLIR_ENABLE_RUNTIME=ON
            -DTTMLIR_ENABLE_RUNTIME_TESTS=ON  # used for emitc testing
            -DTTMLIR_ENABLE_BINDINGS_PYTHON=OFF
            -DTT_RUNTIME_DEBUG=${TTMLIR_RUNTIME_DEBUG}
            -DMLIR_DIR=${TTMLIR_TOOLCHAIN_DIR}/lib/cmake/mlir
            -DTTMLIR_ENABLE_OPMODEL=ON
            -DTTMLIR_ENABLE_EXPLORER=OFF
    )
endif()

install(DIRECTORY ${TTMLIR_INSTALL_PREFIX}/ DESTINATION "${CMAKE_INSTALL_PREFIX}" USE_SOURCE_PERMISSIONS)

### end build tt-mlir ###

### Build TVM ###

add_custom_target(build_tvm ALL
    COMMAND bash -c ${CMAKE_CURRENT_SOURCE_DIR}/tvm/install.sh
    COMMENT "Installing TVM"
    USES_TERMINAL
)

### end build TVM ###
