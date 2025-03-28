// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/python.h>
#include <torch/torch.h>

namespace tt
{
torch::Tensor is_close(
    torch::Tensor a, torch::Tensor b, double rtol = 1e-5, double atol = 1e-9, bool equal_nan = false);
torch::Tensor all_close(torch::Tensor a, torch::Tensor b, double rtol, double atol, bool equal_nan = false);

double max_abs_diff(torch::Tensor& a, torch::Tensor& b);
}  // namespace tt
