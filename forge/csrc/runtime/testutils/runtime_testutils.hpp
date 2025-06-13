// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/python.h>
#include <torch/torch.h>

#include "runtime/tensor.hpp"
#include "tt/runtime/types.h"

namespace tt::runtime_testutils
{

bool test_so(
    std::string so_path,
    std::string func_name,
    std::vector<tt::Tensor>& act_inputs,
    std::vector<tt::Tensor>& consts_and_params,
    std::vector<tt::Tensor>& outputs);

void* open_so(std::string path);

void close_so(void* handle);

std::vector<tt::runtime::Tensor> run_so_program(
    void* so_handle,
    std::string func_name,
    std::vector<tt::Tensor>& inputs,
    std::vector<tt::Tensor>& consts_and_params);

bool compareOuts(std::vector<tt::runtime::Tensor>& lhs, std::vector<tt::runtime::Tensor>& rhs);

}  // namespace tt::runtime_testutils
