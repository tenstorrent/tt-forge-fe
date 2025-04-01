// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "mlir_compiler.hpp"

#include <memory>

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
        forge_property_handler, tt::property::ExecutionDepth::FAILED_TTMLIR_COMPILATION);

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

        tt::property::record_execution_depth(forge_property_handler, tt::property::ExecutionDepth::FAILED_RUNTIME);

        std::string binary_json_str = runtime::Binary(binary).asJson();
        tt::property::record_flatbuffer_details(forge_property_handler, binary_json_str);

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
}  // namespace tt::passes
