// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "mlir_compiler.hpp"

#include <filesystem>
#include <memory>

#include "graph_lib/defines.hpp"
#include "lower_to_mlir.hpp"
#include "mlir_passes.hpp"

// Forge headers
#include "graph_lib/graph.hpp"
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
#include "tt/runtime/types.h"
#include "tt_torch_device/tt_device.hpp"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Target/TTNN/TTNNToFlatbuffer.h"

// Reportify headers
#include "reportify/reportify.hpp"

namespace tt::passes
{
/// Public API for lowering to MLIR, running MLIR passes and generate runtime binary.
runtime::Binary run_mlir_compiler(tt::ForgeGraphModule& module)
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
        mlir::tensor::TensorDialect>();

    mlir::func::registerInlinerExtension(registry);

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

    // Run MLIR registered passes.
    run_mlir_passes(mlir_module);
    tt::log_info(LogMLIRCompiler, "MLIR passes run successfully.");

    mlir_module->dump();

    // save what's dumped to a file named "{name}.mlir"
    reportify::dump_mlir("ttnn", mlir_module->getName()->str(), mlir_module.get());
    std::string moduleName = mlir_module->getName()->str();
    // Generate binary from the MLIR module.
    auto binary = mlir::tt::ttnn::ttnnToFlatbuffer(mlir_module.get());
    tt::log_info(LogMLIRCompiler, "Flatbuffer binary generated successfully.");

    if (binary == nullptr)
    {
        throw std::runtime_error("Failed to generate flatbuffer binary.");
    }
    const std::string folderPath = "bin_and_json/";
    std::filesystem::create_directories(folderPath);
    const std::string binaryFilePath = folderPath + moduleName + ".bin";
    const std::string jsonFilePath = folderPath + moduleName + ".json";

    runtime::Binary binary_obj = runtime::Binary(binary);
    std::string json_str = binary_obj.asJson();
    binary_obj.store(binaryFilePath.c_str());
    tt::log_info(LogMLIRCompiler, "Flatbuffer JSON generated successfully.");

    std::ofstream jsonFile(jsonFilePath);
    if (!jsonFile)
    {
        throw std::runtime_error("Failed to open JSON file for saving: " + jsonFilePath);
    }
    jsonFile << json_str;
    jsonFile.close();

    return binary;
}
}  // namespace tt::passes
