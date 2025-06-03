// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "runtime/test/python_bindings.hpp"

#include "runtime/tensor.hpp"
#include "runtime/test/runtime-test.hpp"

namespace tt
{

void RuntimeTestModule(py::module &m_runtime_test)
{
    m_runtime_test.def(
        "test_so",
        [](std::string so_path,
           std::string func_name,
           std::vector<tt::Tensor> &inputs,
           std::vector<tt::Tensor> &consts_and_params,
           std::vector<tt::Tensor> &outputs)
        { return tt::runtime_test::test_so(so_path, func_name, inputs, consts_and_params, outputs); });
}

}  // namespace tt
