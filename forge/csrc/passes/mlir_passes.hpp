// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
namespace mlir
{
class ModuleOp;
template <typename OpTy>
class OwningOpRef;
}  // namespace mlir

namespace tt::passes
{
/// Public API for running MLIR passes and generating binary.
void run_mlir_passes(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);
}  // namespace tt::passes
