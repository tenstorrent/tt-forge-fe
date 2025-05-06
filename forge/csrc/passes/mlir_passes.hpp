// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>

namespace mlir
{
class ModuleOp;
template <typename OpTy>
class OwningOpRef;
}  // namespace mlir

namespace tt::passes
{

struct MLIRConfig;

enum class MLIROutputKind
{
    Flatbuffer,
    Cpp
};

/// Public API for running MLIR passes (pipeline) depending on the desired output.
template <MLIROutputKind output>
void run_mlir_passes(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module, const std::optional<MLIRConfig> &mlir_config);

}  // namespace tt::passes
