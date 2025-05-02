// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "verif/python_bindings.hpp"

#include "verif/verif_ops.hpp"

namespace tt
{

void VerifModule(py::module &m_verif)
{
    m_verif.def(
        "all_close",
        &all_close,
        "all_close",
        py::arg("a"),
        py::arg("b"),
        py::arg("rtol") = 1e-5,
        py::arg("atol") = 1e-9);
    m_verif.def("max_abs_diff", &max_abs_diff, "max_abs_diff");
    m_verif.def("has_special_values", &has_special_values, "has_special_values");
    m_verif.def("calculate_tensor_pcc", &calculate_tensor_pcc, "calculate_tensor_pcc");
}

}  // namespace tt
