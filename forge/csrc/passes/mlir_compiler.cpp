// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "mlir_compiler.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
namespace fs = std::filesystem;

#include "graph_lib/defines.hpp"
#include "lower_to_mlir.hpp"
#include "mlir_passes.hpp"

// Forge headers
#include "graph_lib/graph.hpp"
#include "shared_utils/forge_property_utils.hpp"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#include "graph_lib/node_types.hpp"
#pragma clang diagnostic pop

// MLIR headers
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/IR/BuiltinOps.h"
#include "utils/logger.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-local-typedef"
#include "mlir/InitAllDialects.h"
#pragma clang diagnostic pop

// TTMLIR headers
#include "compile_so.hpp"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "tt/runtime/types.h"
#include "tt_torch_device/tt_device.hpp"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/Transforms/TTNNToCpp.h"
#include "ttmlir/Target/TTNN/TTNNToFlatbuffer.h"

// Reportify headers
#include "reportify/reportify.hpp"

namespace tt::passes
{

// Template function to run the MLIR compiler pipeline, depending on the desired output.
template <MLIROutputKind output>
auto run_mlir_compiler_generic(tt::ForgeGraphModule& module, const std::optional<py::object>& forge_property_handler)
{
    // Register all the required dialects.
    mlir::DialectRegistry registry;

    registry.insert<
        mlir::tt::TTDialect,
        mlir::tt::ttir::TTIRDialect,
        mlir::tt::ttnn::TTNNDialect,
        mlir::arith::ArithDialect,
        mlir::func::FuncDialect,
        mlir::ml_program::MLProgramDialect,
        mlir::tensor::TensorDialect,
        mlir::LLVM::LLVMDialect>();

    mlir::func::registerInlinerExtension(registry);

    // Register the LLVM dialect inliner extension
    mlir::LLVM::registerInlinerInterface(registry);

    // Create a context with all registered dialects.
    mlir::MLIRContext context(registry);

#ifdef DEBUG
    // Context setting to have mlir print out stacktrace whenever errors occur
    context.printStackTraceOnDiagnostic(true);
#endif

    // Load all available dialects
    context.loadAllAvailableDialects();

    // Generate MLIR from the Forge graph.
    mlir::OwningOpRef<mlir::ModuleOp> mlir_module = lower_to_mlir(module, context);

    tt::property::record_execution_depth(
        tt::property::ExecutionDepth::FAILED_TTMLIR_COMPILATION, forge_property_handler);

    // Run MLIR pipeline.
    run_mlir_passes<output>(mlir_module);

    tt::log_info(LogMLIRCompiler, "MLIR passes run successfully.");

    mlir_module->dump();

    if constexpr (output == MLIROutputKind::Flatbuffer)
    {
        // Save generated ttnn module to a file named "{name}.mlir".
        reportify::dump_mlir("ttnn", mlir_module->getName()->str(), mlir_module.get());

        // Generate binary from the MLIR module.
        auto binary = mlir::tt::ttnn::ttnnToFlatbuffer(mlir_module.get());
        tt::log_info(LogMLIRCompiler, "Flatbuffer binary generated successfully.");

        if (binary == nullptr)
        {
            throw std::runtime_error("Failed to generate flatbuffer binary.");
        }

        tt::property::record_execution_depth(tt::property::ExecutionDepth::FAILED_RUNTIME, forge_property_handler);

        std::string binary_json_str = runtime::Binary(binary).asJson();
        tt::property::record_flatbuffer_details(binary_json_str, forge_property_handler);

        return binary;
    }
    else if constexpr (output == MLIROutputKind::Cpp)
    {
        std::string cpp_source;
        llvm::raw_string_ostream rso(cpp_source);

        log_info(LogMLIRCompiler, "Generating C++ code from MLIR module.");
        auto res = mlir::emitc::translateToCpp(mlir_module.get(), rso);
        if (mlir::failed(res))
        {
            throw std::runtime_error("Failed to generate C++ code.");
        }

        rso.flush();

        tt::log_info(LogMLIRCompiler, "C++ code generated successfully.");
        return cpp_source;
    }
    else if constexpr (output == MLIROutputKind::SharedObject)
    {
        std::string cpp_source;
        llvm::raw_string_ostream rso(cpp_source);

        {
            std::cout << "PRINTING ENVS RMCG" << std::endl;
            std::cout << "I'm in run_mlir_compiler_generic<MLIROutputKind::SharedObject>" << std::endl;
            const char* var_value;

            var_value = std::getenv("TT_METAL_HOME");
            std::cout << "  TT_METAL_HOME"
                      << " environment variable: " << (var_value != nullptr ? var_value : "not set") << std::endl;

            var_value = std::getenv("CMAKE_INSTALL_PREFIX");
            std::cout << "  CMAKE_INSTALL_PREFIX"
                      << " environment variable: " << (var_value != nullptr ? var_value : "not set") << std::endl;

            var_value = std::getenv("TT_MLIR_HOME");
            std::cout << "  TT_MLIR_HOME" << " environment variable: " << (var_value != nullptr ? var_value : "not set")
                      << std::endl;

            var_value = std::getenv("FORGE_HOME");
            std::cout << "  FORGE_HOME" << " environment variable: " << (var_value != nullptr ? var_value : "not set")
                      << std::endl;

            std::cout << "  PRINTING FROM " << __FILE__ << std::endl;
        }

        log_info(LogMLIRCompiler, "Generating a shared object from MLIR module.");
        auto res = mlir::emitc::translateToCpp(mlir_module.get(), rso);
        if (mlir::failed(res))
        {
            throw std::runtime_error("Failed to generate C++ code.");
        }

        rso.flush();

        tt::log_info(LogMLIRCompiler, "C++ code for SharedObject generated successfully.");

        const char* TT_METAL_HOME = std::getenv("TT_METAL_HOME");
        tt::log_info(LogMLIRCompiler, "TT_METAL_HOME: {}", TT_METAL_HOME);
        const char* FORGE_HOME = std::getenv("FORGE_HOME");
        tt::log_info(LogMLIRCompiler, "FORGE_HOME: {}", FORGE_HOME);
        if (TT_METAL_HOME == nullptr)
        {
            tt::log_info(LogMLIRCompiler, "TT_METAL_HOME is not set.");
            std::cout << "FAILED AT 1" << std::endl;
            throw std::runtime_error("TT_METAL_HOME environment variable is not set.");
        }
        if (FORGE_HOME == nullptr)
        {
            tt::log_info(LogMLIRCompiler, "FORGE_HOME is not set.");
            std::cout << "FAILED AT 2" << std::endl;
            throw std::runtime_error("FORGE_HOME environment variable is not set.");
        }

        tt::log_info(LogMLIRCompiler, "before in_wheel_path");
        const fs::path in_wheel_path = fs::path(FORGE_HOME) / "forge/tt-metal";
        tt::log_info(LogMLIRCompiler, "after in_wheel_path");
        const fs::path in_source_path =
            fs::path(FORGE_HOME).parent_path() / "third_party/tt-mlir/third_party/tt-metal/src/tt-metal";
        tt::log_info(LogMLIRCompiler, "after in_source_path");

        std::cout << "in_wheel_path: " << in_wheel_path << std::endl;
        std::cout << "in_source_path: " << in_source_path << std::endl;

        if (!fs::exists(in_wheel_path) && !fs::exists(in_source_path))
        {
            tt::log_info(LogMLIRCompiler, "Neither tt-metal wheel nor source path exists.");
            std::cout << "FAILED AT 3" << std::endl;
            throw std::runtime_error("Neither tt-metal wheel nor source path exists.");
        }

        fs::path metal_src_dir;
        fs::path metal_lib_dir;
        fs::path standalone_dir;
        if (fs::exists(in_wheel_path))
        {
            tt::log_info(LogMLIRCompiler, "in_wheel_path exists");
            std::cout << "in_wheel_path exists" << std::endl;
            std::cout << in_wheel_path << std::endl;
            std::cout << std::string(TT_METAL_HOME) << std::endl;
            assert(in_wheel_path == std::string(TT_METAL_HOME));
            std::cout << "Using in_wheel_path: " << in_wheel_path << std::endl;
            metal_src_dir = fs::path(std::string(TT_METAL_HOME));
            metal_lib_dir = fs::path(std::string(FORGE_HOME)) / "forge/lib";
            standalone_dir = fs::path(std::string(FORGE_HOME)) / "forge/tools/ttnn-standalone";
        }
        else if (fs::exists(in_source_path))
        {
            tt::log_info(LogMLIRCompiler, "in_source_path exists");
            std::cout << "in_source_path exists" << std::endl;
            std::cout << in_source_path << std::endl;
            std::cout << std::string(TT_METAL_HOME) << std::endl;
            assert(in_source_path == std::string(TT_METAL_HOME));
            std::cout << "Using in_source_path: " << in_source_path << std::endl;
            metal_src_dir = fs::path(std::string(TT_METAL_HOME));
            metal_lib_dir = fs::path(std::string(TT_METAL_HOME)).parent_path() / "tt-metal-build/lib";
            standalone_dir =
                fs::path(std::string(FORGE_HOME)).parent_path() / "third_party/tt-mlir/tools/ttnn-standalone";
        }

        tt::log_info(LogMLIRCompiler, "before compile_cpp_to_so");
        std::string soPathStr = compile_cpp_to_so(
            cpp_source, "/tmp/", metal_src_dir.string(), metal_lib_dir.string(), standalone_dir.string());

        return soPathStr;
    }
}

runtime::Binary run_mlir_compiler(tt::ForgeGraphModule& module, const std::optional<py::object>& forge_property_handler)
{
    return run_mlir_compiler_generic<MLIROutputKind::Flatbuffer>(module, forge_property_handler);
}

std::string run_mlir_compiler_to_cpp(
    tt::ForgeGraphModule& module, const std::optional<py::object>& forge_property_handler)
{
    return run_mlir_compiler_generic<MLIROutputKind::Cpp>(module, forge_property_handler);
}

std::string run_mlir_compiler_to_shared_object(
    tt::ForgeGraphModule& module, const std::optional<py::object>& forge_property_handler)
{
    return run_mlir_compiler_generic<MLIROutputKind::SharedObject>(module, forge_property_handler);
}
}  // namespace tt::passes
