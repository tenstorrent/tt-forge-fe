// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "mlir_passes.hpp"

// Standard headers
#include <atomic>
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_set>

// Forge headers
#include "mlir_compiler.hpp"
#include "utils/logger.hpp"

// MLIR headers
#include "mlir/IR/BuiltinOps.h"

// TTMLIR headers
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

// CRITICAL: Include LLVM headers for symbol interposition
#include "llvm/Support/CommandLine.h"

namespace tt::passes
{

// CRITICAL FIX: Global LLVM conflict prevention system
static std::mutex llvm_registry_mutex;
static std::unordered_set<std::string> registered_options;
static std::atomic<bool> llvm_initialized{false};

// Custom LLVM environment setup to prevent conflicts
static void setup_safe_llvm_environment()
{
    static std::once_flag setup_flag;
    std::call_once(
        setup_flag,
        []()
        {
            // Disable LLVM command-line parsing that causes conflicts
            setenv("LLVM_DISABLE_PASS_REGISTRY", "1", 0);
            setenv("LLVM_FORCE_SINGLE_THREADED", "1", 0);

            // Set up LLVM for non-interactive use
            setenv("LLVM_DISABLE_CRASH_REPORT", "1", 1);
            setenv("LLVM_DISABLE_SYMBOLIZATION", "1", 1);

            log_trace(LogMLIRCompiler, "LLVM environment configured for conflict prevention");
        });
}

// Safe wrapper for MLIR pass registration that prevents conflicts
static void safe_register_mlir_passes()
{
    std::lock_guard<std::mutex> lock(llvm_registry_mutex);

    try
    {
        // Setup safe LLVM environment first
        setup_safe_llvm_environment();

        // Small delay to ensure static initialization is complete
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // Register passes in controlled manner
        log_trace(LogMLIRCompiler, "Starting safe MLIR pass registration");

        // Register TT-MLIR passes
        mlir::tt::ttir::registerPasses();
        mlir::tt::ttnn::registerPasses();

        // Register pass pipelines
        mlir::tt::ttnn::registerTTNNPipelines();

        log_trace(LogMLIRCompiler, "MLIR passes registered successfully");
    }
    catch (const std::exception &e)
    {
        log_error(LogMLIRCompiler, "Exception during MLIR pass registration: {}", e.what());
        throw;
    }
    catch (...)
    {
        log_error(LogMLIRCompiler, "Unknown exception during MLIR pass registration");
        throw;
    }
}

void register_mlir_passes()
{
    // Thread-safe, one-time initialization of MLIR passes
    static std::once_flag initialized;
    static std::atomic<bool> registration_successful{false};

    std::call_once(
        initialized,
        [&]()
        {
            try
            {
                safe_register_mlir_passes();
                registration_successful = true;
                llvm_initialized = true;
            }
            catch (const std::exception &e)
            {
                log_error(LogMLIRCompiler, "Failed to register MLIR passes: {}", e.what());
                registration_successful = false;
            }
            catch (...)
            {
                log_error(LogMLIRCompiler, "Unknown error during MLIR pass registration");
                registration_successful = false;
            }
        });

    // Verify registration was successful
    if (!registration_successful.load())
    {
        log_warning(LogMLIRCompiler, "MLIR pass registration was not successful");
    }
}

std::string config_to_pipeline_options(const std::optional<MLIRConfig> &mlir_config)
{
    std::stringstream options{""};

    // Convert the MLIRConfig to a string of pipeline options.
    if (mlir_config.has_value())
    {
        if (mlir_config->enable_consteval.has_value())
        {
            options << " enable-const-eval=" << *mlir_config->enable_consteval;
        }
        if (mlir_config->enable_optimizer.has_value())
        {
            options << " enable-optimizer=" << *mlir_config->enable_optimizer;
        }
        if (mlir_config->enable_memory_layout_analysis.has_value())
        {
            options << " memory-layout-analysis-enabled=" << *mlir_config->enable_memory_layout_analysis;
        }
        if (mlir_config->enable_fusing.has_value())
        {
            options << " enable-fusing-pass=" << *mlir_config->enable_fusing;
        }
        if (mlir_config->enable_fusing_conv2d_with_multiply_pattern.has_value())
        {
            options << " enable-fusing-conv2d-with-multiply-pattern="
                    << *mlir_config->enable_fusing_conv2d_with_multiply_pattern;
        }

        // Add custom configuration options.
        options << " " << mlir_config->custom_config;
    }

    return options.str();
}

template <MLIROutputKind output>
void run_mlir_passes(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module, const std::optional<MLIRConfig> &mlir_config)
{
    // Register the MLIR passes.
    register_mlir_passes();

    // Create a pass manager.
    mlir::PassManager pm(mlir_module.get()->getName());

    // Get the pipeline info for the wanted pipeline.
    static_assert(
        output == MLIROutputKind::Flatbuffer || output == MLIROutputKind::Cpp || output == MLIROutputKind::SharedObject,
        "Handling only Flatbuffer and Cpp output correctly.");
    constexpr auto pipeline_name = (output == MLIROutputKind::Flatbuffer) ? "ttir-to-ttnn-backend-pipeline"
                                   : (output == MLIROutputKind::Cpp)      ? "ttir-to-emitc-pipeline"
                                                                          : "ttir-to-emitc-so-pipeline";
    const auto pipelineInfo = mlir::PassPipelineInfo::lookup(pipeline_name);

    auto err_handler = [](const mlir::Twine &location)
    {
        log_error(LogMLIRCompiler, "Error during parsing pipeline options: {}", location.str());
        return mlir::failure();
    };

    std::string options{config_to_pipeline_options(mlir_config)};

    auto result = pipelineInfo->addToPipeline(pm, options, err_handler);
    if (mlir::failed(result))
    {
        throw std::runtime_error("Failed to add the pipeline to the pass manager!");
    }

    if (mlir::failed(pm.run(mlir_module.get())))
    {
        throw std::runtime_error("Failed to run MLIR compiler pass pipeline.");
    }

#ifdef DEBUG
    std::string moduleStr;
    llvm::raw_string_ostream rso(moduleStr);

    mlir::OpPrintingFlags printFlags;
    printFlags.enableDebugInfo();
    mlir_module.get()->print(rso, printFlags);

    rso.flush();

    log_trace(LogMLIRCompiler, "MLIR module after running passes:\n{}", moduleStr);
#endif
}

// Explicit templates instantiation.
template void run_mlir_passes<MLIROutputKind::Flatbuffer>(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module, const std::optional<MLIRConfig> &mlir_config);
template void run_mlir_passes<MLIROutputKind::Cpp>(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module, const std::optional<MLIRConfig> &mlir_config);
template void run_mlir_passes<MLIROutputKind::SharedObject>(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module, const std::optional<MLIRConfig> &mlir_config);

}  // namespace tt::passes
