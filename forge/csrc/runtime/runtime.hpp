// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/python.h>
#include <torch/torch.h>

#include "tensor.hpp"
#include "tt/runtime/types.h"

namespace tt
{

// Entry point for invoking tt-mlir runtime and running the binary on the device.
std::vector<torch::Tensor> run_binary(
    runtime::Binary& binary, int program_idx, std::vector<torch::Tensor> const& inputs);

std::pair<std::vector<tt::Tensor>, std::vector<torch::Tensor>> run_binary_v2(
    runtime::Binary& binary,
    int program_idx,
    std::vector<torch::Tensor> const& act_inputs,
    std::vector<tt::Tensor>& persistent_inputs);

// Helper function to run the binary from the file - might be useful for testing/debugging.
std::vector<torch::Tensor> run_binary_from_file(
    std::string const& filename, int program_idx, std::vector<torch::Tensor> const& inputs);

}  // namespace tt
