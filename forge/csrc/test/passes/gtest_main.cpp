// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <dlfcn.h>
#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "passes/mlir_passes.hpp"
#include "test/common.hpp"

// Define RTLD_DEEPBIND if not available
#ifndef RTLD_DEEPBIND
#define RTLD_DEEPBIND 0x8
#endif

namespace py = pybind11;

void setup_llvm_environment()
{
    // LLVM crash reporting and debugging
    setenv("LLVM_DISABLE_CRASH_REPORT", "1", 1);
    setenv("LLVM_ENABLE_DUMP", "0", 1);
    setenv("LLVM_DISABLE_SYMBOLIZATION", "1", 1);

    // TensorFlow optimizations
    setenv("TF_ENABLE_ONEDNN_OPTS", "0", 1);

    // Threading control
    setenv("OMP_NUM_THREADS", "1", 1);
    setenv("MKL_NUM_THREADS", "1", 1);

    // Python environment isolation
    setenv("PYTHONDONTWRITEBYTECODE", "1", 1);
    setenv("PYTHONHASHSEED", "0", 1);
    setenv("PYTHONHOME", "", 1);
    setenv("PYTHONPATH", "", 1);
}

bool preload_critical_libraries()
{
    // List of libraries to preload in specific order to avoid conflicts
    const char* libraries[] = {
        "libcrypto.so.3",  // OpenSSL crypto (load early)
        "libssl.so.3",     // OpenSSL SSL
        "libblas.so.3",    // BLAS
        "liblapack.so.3",  // LAPACK
        nullptr};

    // Use RTLD_NOW | RTLD_GLOBAL to resolve all symbols immediately and make them globally available
    for (const char** lib = libraries; *lib != nullptr; ++lib)
    {
        void* handle = dlopen(*lib, RTLD_NOW | RTLD_GLOBAL);
        if (handle)
        {
            std::cout << "Successfully preloaded: " << *lib << std::endl;
        }
        else
        {
            std::cout << "Could not preload " << *lib << ": " << dlerror() << std::endl;
        }
    }

    // CRITICAL: Preload TensorFlow with RTLD_DEEPBIND to force symbol isolation
    const char* tensorflow_lib =
        "/opt/ttforge-toolchain/venv/lib/python3.10/site-packages/tensorflow/python/platform/../../"
        "libtensorflow_cc.so.2";
    void* tf_handle = dlopen(tensorflow_lib, RTLD_NOW | RTLD_DEEPBIND);
    if (tf_handle)
    {
        std::cout << "Successfully preloaded TensorFlow with RTLD_DEEPBIND for symbol isolation" << std::endl;
    }
    else
    {
        std::cout << "Could not preload TensorFlow with DEEPBIND: " << dlerror() << std::endl;
        // Try regular loading as fallback
        tf_handle = dlopen(tensorflow_lib, RTLD_NOW);
        if (tf_handle)
        {
            std::cout << "Loaded TensorFlow without DEEPBIND as fallback" << std::endl;
        }
        else
        {
            std::cout << "Could not preload TensorFlow at all: " << dlerror() << std::endl;
        }
    }

    return true;
}

bool setup_memory_protection()
{
    // Increase stack size for deep Python import chains
    struct rlimit rl;
    if (getrlimit(RLIMIT_STACK, &rl) == 0)
    {
        rl.rlim_cur = std::min(rl.rlim_max, (rlim_t)(16 * 1024 * 1024));  // 16MB stack
        setrlimit(RLIMIT_STACK, &rl);
    }

    // Enable core dumps for debugging
    rl.rlim_cur = rl.rlim_max = RLIM_INFINITY;
    setrlimit(RLIMIT_CORE, &rl);

    return true;
}

bool safe_python_initialization()
{
    try
    {
        std::cout << "Initializing Python interpreter..." << std::endl;

        // Initialize with minimal configuration
        py::initialize_interpreter(false);

        std::cout << "Python interpreter initialized successfully" << std::endl;

        // Test basic Python functionality
        try
        {
            py::gil_scoped_acquire acquire;

            // Try to import sys first
            py::module_ sys = py::module_::import("sys");
            std::cout << "Successfully imported sys module" << std::endl;

            // Try to import hashlib
            try
            {
                py::module_ hashlib = py::module_::import("hashlib");
                std::cout << "Successfully imported hashlib module" << std::endl;
            }
            catch (const std::exception& e)
            {
                std::cout << "Failed to import hashlib: " << e.what() << std::endl;
                return false;
            }

            // Try to import numpy
            try
            {
                py::module_ numpy = py::module_::import("numpy");
                std::cout << "Successfully imported numpy module" << std::endl;
            }
            catch (const std::exception& e)
            {
                std::cout << "Failed to import numpy: " << e.what() << std::endl;
            }

            acquire.disarm();
        }
        catch (const std::exception& e)
        {
            std::cout << "Failed during Python module testing: " << e.what() << std::endl;
            return false;
        }

        return true;
    }
    catch (const std::exception& e)
    {
        std::cout << "Failed to initialize Python interpreter: " << e.what() << std::endl;
        return false;
    }
}

int main(int argc, char** argv)
{
    std::cout << "Setting up test environment..." << std::endl;

    // Step 1: Setup environment variables
    setup_llvm_environment();

    // Step 2: Preload critical libraries to avoid conflicts
    preload_critical_libraries();

    // Step 3: Setup memory protection
    setup_memory_protection();

    // Step 4: Initialize Google Test
    testing::InitGoogleTest(&argc, argv);

    // Step 5: Setup GraphTest environment
    GraphTestFlags flags = GraphTestFlags::from_cli_args(argc, argv);
    ::testing::AddGlobalTestEnvironment(new GraphTestFlagsEnvironment(flags));

    // Step 6: Safe Python initialization
    if (!safe_python_initialization())
    {
        std::cerr << "Failed to safely initialize Python. Aborting tests." << std::endl;
        return 1;
    }

    // Step 7: Run tests with GIL properly managed
    int result;
    try
    {
        py::gil_scoped_acquire acquire;
        acquire.disarm();

        std::cout << "Running tests..." << std::endl;
        result = RUN_ALL_TESTS();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception during test execution: " << e.what() << std::endl;
        result = 1;
    }

    // Step 8: Cleanup
    try
    {
        py::finalize_interpreter();
        std::cout << "Python interpreter finalized successfully" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cout << "Warning: Exception during Python finalization: " << e.what() << std::endl;
    }

    return result;
}
