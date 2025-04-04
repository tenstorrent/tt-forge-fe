// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/python.h>
#include <torch/torch.h>

namespace tt
{
double max_abs_diff(torch::Tensor& a, torch::Tensor& b);
bool all_close(torch::Tensor a, torch::Tensor b, double rtol = 1e-5, double atol = 1e-9, bool equal_nan = false);

bool has_special_values(torch::Tensor& a);

double calculate_tensor_pcc(torch::Tensor& a, torch::Tensor& b);
}  // namespace tt
