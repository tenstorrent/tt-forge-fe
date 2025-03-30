// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "verif/python_bindings.hpp"

#include "verif/verif_ops.hpp"

namespace tt
{

void VerifModule(py::module &m_verif)
{
    m_verif.def(
        "is_close",
        &is_close,
        "is_close",
        py::arg("a"),
        py::arg("b"),
        py::arg("rtol") = 1e-5,
        py::arg("atol") = 1e-9,
        py::arg("equal_nan") = false);
    m_verif.def("all_close", &all_close, "all_close");
    m_verif.def("max_abs_diff", &max_abs_diff, "max_abs_diff");
    m_verif.def("has_special_values", &has_special_values, "has_special_values");
}

}  // namespace tt
