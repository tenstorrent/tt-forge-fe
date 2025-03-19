// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#include <pybind11/pybind11.h>
#pragma clang diagnostic pop

#include "tt/runtime/types.h"

namespace py = pybind11;

namespace tt
{
class ForgeGraphModule;
}

namespace tt::passes
{
/// Public API for running MLIR passes and generating binary.
runtime::Binary run_mlir_compiler(
    tt::ForgeGraphModule& module, const std::optional<py::object>& forge_property_handler = std::nullopt);

/// Public API for lowering to MLIR, running MLIR passes and generate C++ code.
std::string run_mlir_compiler_to_cpp(
    tt::ForgeGraphModule& module, const std::optional<py::object>& forge_property_handler = std::nullopt);
}  // namespace tt::passes
