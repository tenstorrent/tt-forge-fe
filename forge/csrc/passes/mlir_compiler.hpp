// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#include <pybind11/pybind11.h>
#pragma clang diagnostic pop

#include "nlohmann/json_fwd.hpp"
#include "tt/runtime/types.h"

namespace py = pybind11;

namespace tt
{
class ForgeGraphModule;
}

namespace tt::passes
{

/// Struct to hold the configuration for the MLIR compiler.
struct MLIRConfig
{
    std::optional<bool> enable_consteval = std::nullopt;
    std::optional<bool> enable_optimizer = std::nullopt;
    std::optional<bool> enable_memory_layout_analysis = std::nullopt;

    // Custom configuration string for the MLIR compiler.
    std::string custom_config = "";

    MLIRConfig& set_enable_consteval(bool enable)
    {
        enable_consteval = enable;
        return *this;
    }

    MLIRConfig& set_enable_optimizer(bool enable)
    {
        enable_optimizer = enable;
        return *this;
    }

    MLIRConfig& set_enable_memory_layout_analysis(bool enable)
    {
        enable_memory_layout_analysis = enable;
        return *this;
    }

    // Set custom configuration string for the MLIR compiler.
    // This can be used to pass any additional configuration options to the MLIR compiler - not covered by the existing
    // knobs.
    MLIRConfig& set_custom_config(const std::string& config)
    {
        custom_config = config;
        return *this;
    }
};

void to_json(nlohmann::json& j, const MLIRConfig& p);
void from_json(const nlohmann::json& j, MLIRConfig& p);

/// Public API for running MLIR passes and generating binary.
runtime::Binary run_mlir_compiler(
    tt::ForgeGraphModule& module,
    const std::optional<MLIRConfig>& mlir_config = std::nullopt,
    const std::optional<py::object>& forge_property_handler = std::nullopt);

/// Public API for lowering to MLIR, running MLIR passes and generate C++ code.
std::string run_mlir_compiler_to_cpp(
    tt::ForgeGraphModule& module,
    const std::optional<MLIRConfig>& mlir_config = std::nullopt,
    const std::optional<py::object>& forge_property_handler = std::nullopt);

}  // namespace tt::passes
