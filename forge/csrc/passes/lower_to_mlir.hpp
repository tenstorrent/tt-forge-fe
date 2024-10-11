// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace tt
{
class ForgeGraphModule;
}

namespace mlir
{
class MLIRContext;
class ModuleOp;
template <typename OpTy>
class OwningOpRef;
}  // namespace mlir

namespace tt::passes
{
// Public API for generating MLIR from a Forge module (set of graphs).
mlir::OwningOpRef<mlir::ModuleOp> lower_to_mlir(tt::ForgeGraphModule& module, mlir::MLIRContext& context);
}  // namespace tt::passes
