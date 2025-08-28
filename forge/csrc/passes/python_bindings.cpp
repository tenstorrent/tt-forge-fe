// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "graph_lib/defines.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node_types.hpp"
#include "lower_to_forge/common.hpp"
#include "passes/decomposing_context.hpp"
#include "python_bindings_common.hpp"
#include "torch/extension.h"  // Needed for c++ to/from python type conversion.
#include "torch/torch.h"

namespace tt
{

void PassesModule(py::module &m_passes)
{
    py::class_<tt::DecomposingContext>(m_passes, "DecomposingContext")
        .def("node_name", &tt::DecomposingContext::get_node_name)
        .def("is_training_enabled", &tt::DecomposingContext::is_training_enabled)
        .def(
            "op",
            [](tt::DecomposingContext &self,
               std::variant<std::string, py::object> const &type,
               std::vector<NodeContext> const &operands,
               bool copy_tms = true,
               bool dont_decompose = false,
               bool optimize_hoist = false,
               DataFormat output_df = DataFormat::Invalid)
            {
                if (std::holds_alternative<std::string>(type))
                {
                    return self.op(
                        ops::Op(std::get<std::string>(type)),
                        operands,
                        copy_tms,
                        dont_decompose,
                        optimize_hoist,
                        output_df);
                }
                else
                {
                    auto const &op_type = std::get<py::object>(type).attr("op_type").cast<ops::Op>();
                    return self.op(op_type, operands, copy_tms, dont_decompose, optimize_hoist, output_df);
                }
            },
            py::arg("type"),
            py::arg("operands"),
            py::arg("copy_tms") = true,
            py::arg("dont_decompose") = false,
            py::arg("optimize_hoist") = false,
            py::arg("output_df") = DataFormat::Invalid)
        .def("fuse", &tt::DecomposingContext::fuse, py::arg("operand"), py::arg("producer_output_port_id") = 0)
        .def(
            "tensor",
            [](tt::DecomposingContext &self, py::object tensor) { return self.tensor(py::cast<at::Tensor>(tensor)); },
            py::arg("tensor"))
        .def(
            "get_pytorch_tensor",
            [](tt::DecomposingContext &self, graphlib::NodeContext const &node)
            {
                graphlib::ConstantInputNode *cnode =
                    dynamic_cast<graphlib::ConstantInputNode *>(self.get_graph()->node_by_id(node.id));
                TT_ASSERT(cnode && cnode->is_tensor(), "Only use for ConstantInputNode of type tensor");
                return borrow_shared_py_object(cnode->tensor());
            })
        .def(
            "get_operands",
            [](tt::DecomposingContext &self, graphlib::NodeContext const &node) { return self.get_operands(node); })
        .def(
            "get_compiler_cfg",
            [](tt::DecomposingContext &self) { return borrow_shared_py_object(self.get_compiler_cfg()); });
}

}  // namespace tt
