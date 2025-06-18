// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "graph_lib/defines.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node_types.hpp"
#include "lower_to_forge/common.hpp"
#include "passes/decomposing_context.hpp"
#include "passes/lowering_context.hpp"
#include "python_bindings_common.hpp"
#include "shared_utils/sparse_matmul_utils.hpp"

namespace tt
{

static bool has_newstyle_interface(std::string const &op_name)
{
    py::object eval_module = py::module_::import("forge.op.eval.forge");
    return eval_module.attr("has_newstyle_interface")(op_name).cast<bool>();
}

void PassesModule(py::module &m_passes)
{
    py::class_<tt::LoweringContext>(m_passes, "LoweringContext")
        .def(
            "op",
            [](tt::LoweringContext &self,
               std::variant<std::string, py::object> const &type,
               std::vector<NodeContext> const &operands,
               std::vector<graphlib::OpType::Attr> const &attrs = {},
               ForgeOpAttrs const &forge_attrs = {},
               std::string const &tag = "",
               int tile_height = graphlib::Shape::FORGE_TILE_DIM,
               int tile_width = graphlib::Shape::FORGE_TILE_DIM)
            {
                if (std::holds_alternative<std::string>(type))
                {
                    TT_LOG_ASSERT(
                        not has_newstyle_interface(std::get<std::string>(type)),
                        "Error lowering a type with old OpType interface, expects new OpType interface {}",
                        std::get<std::string>(type));
                    return self.op(
                        graphlib::OpType(std::get<std::string>(type), attrs, forge_attrs),
                        operands,
                        tag,
                        tile_height,
                        tile_width);
                }
                else
                {
                    TT_ASSERT(attrs.size() == 0, "Illegal mixing of API modes");
                    TT_ASSERT(forge_attrs.size() == 0, "Illegal mixing of API modes");
                    auto const &op_type = std::get<py::object>(type).attr("op_type").cast<graphlib::OpType>();
                    return self.op(op_type, operands, tag, tile_height, tile_width);
                }
            },
            py::arg("type"),
            py::arg("operands"),
            py::arg("attrs") = std::vector<int>{},
            py::arg("forge_attrs") = ForgeOpAttrs{},
            py::arg("tag") = "",
            py::arg("tile_height") = graphlib::Shape::FORGE_TILE_DIM,
            py::arg("tile_width") = graphlib::Shape::FORGE_TILE_DIM)
        .def(
            "tm",
            [](tt::LoweringContext &self,
               std::variant<std::string, py::object> const &type,
               NodeContext const &operand,
               std::vector<graphlib::OpType::Attr> const &attrs = {},
               ForgeOpAttrs const &forge_attrs = {})
            {
                if (std::holds_alternative<std::string>(type))
                {
                    TT_LOG_ASSERT(
                        not has_newstyle_interface(std::get<std::string>(type)),
                        "Error lowering a type with old OpType interface, expects new OpType interface {}",
                        std::get<std::string>(type));
                    return self.tm(graphlib::OpType(std::get<std::string>(type), attrs, forge_attrs), operand);
                }
                else
                {
                    TT_ASSERT(attrs.size() == 0, "Illegal mixing of API modes");
                    TT_ASSERT(forge_attrs.size() == 0, "Illegal mixing of API modes");
                    auto const &op_type = std::get<py::object>(type).attr("op_type").cast<graphlib::OpType>();
                    return self.tm(op_type, operand);
                }
            },
            py::arg("type"),
            py::arg("operand"),
            py::arg("attrs") = std::vector<int>{},
            py::arg("forge_attrs") = ForgeOpAttrs{})
        .def("shape", &tt::LoweringContext::shape, py::arg("node"), py::arg("use_new_graph") = false)
        .def(
            "set_broadcast_dim",
            &tt::LoweringContext::set_broadcast_dim,
            py::arg("src"),
            py::arg("dest"),
            py::arg("dim"),
            py::arg("factor"),
            py::arg("explicit_bcast") = false)
        .def("set_output_df", &tt::LoweringContext::set_output_df)
        .def("set_runtime_tensor_transform", &tt::LoweringContext::set_runtime_tensor_transform)
        .def("constant", &tt::LoweringContext::constant)
        .def(
            "tensor",
            [](tt::LoweringContext &self, py::object tensor, DataFormat df)
            {
                return self.tensor(
                    make_shared_py_object(tensor),
                    graphlib::Shape::create(tensor.attr("shape").cast<std::vector<std::uint32_t>>()),
                    df);
            },
            py::arg("tensor"),
            py::arg("df") = DataFormat::Invalid)
        .def(
            "tensor_with_sparse_forge",
            [](tt::LoweringContext &self, py::object tensor, tt::sparse::SparseFORGE sparse_forge, DataFormat df)
            {
                return self.tensor_with_blob(
                    make_shared_py_object(tensor),
                    graphlib::Shape::create(tensor.attr("shape").cast<std::vector<std::uint32_t>>()),
                    sparse_forge,
                    df);
            },
            py::arg("tensor"),
            py::arg("sparse_forge"),
            py::arg("df") = DataFormat::Invalid)
        .def(
            "get_pytorch_tensor",
            [](tt::LoweringContext &self, graphlib::NodeContext const &node)
            {
                graphlib::ConstantInputNode *cnode =
                    dynamic_cast<graphlib::ConstantInputNode *>(self.get_old_graph()->node_by_id(node.id));
                TT_ASSERT(cnode && cnode->is_tensor(), "Only use for ConstantInputNode of type tensor");
                return borrow_shared_py_object(cnode->tensor());
            })
        .def("forge_shape", &tt::LoweringContext::forge_shape);

    py::class_<tt::DecomposingContext>(m_passes, "DecomposingContext")
        .def("node_name", &tt::DecomposingContext::get_node_name)
        .def("is_training_enabled", &tt::DecomposingContext::is_training_enabled)
        .def(
            "op",
            [](tt::DecomposingContext &self,
               std::variant<std::string, py::object> const &type,
               std::vector<NodeContext> const &operands,
               std::vector<graphlib::OpType::Attr> const &attrs = {},
               bool copy_tms = true,
               bool dont_decompose = false,
               bool optimize_hoist = false,
               DataFormat output_df = DataFormat::Invalid)
            {
                if (std::holds_alternative<std::string>(type))
                {
                    TT_LOG_ASSERT(
                        not has_newstyle_interface(std::get<std::string>(type)),
                        "Error decomposing a type with old OpType interface, expects new OpType interface {}",
                        std::get<std::string>(type));
                    return self.op(
                        graphlib::OpType(std::get<std::string>(type), attrs),
                        operands,
                        copy_tms,
                        dont_decompose,
                        optimize_hoist,
                        output_df);
                }
                else
                {
                    TT_ASSERT(attrs.size() == 0, "Illegal mixing of API modes");
                    auto const &op_type = std::get<py::object>(type).attr("op_type").cast<graphlib::OpType>();
                    return self.op(op_type, operands, copy_tms, dont_decompose, optimize_hoist, output_df);
                }
            },
            py::arg("type"),
            py::arg("operands"),
            py::arg("attrs") = std::vector<int>{},
            py::arg("copy_tms") = true,
            py::arg("dont_decompose") = false,
            py::arg("optimize_hoist") = false,
            py::arg("output_df") = DataFormat::Invalid)
        .def(
            "op_with_named_attrs",
            [](tt::DecomposingContext &self,
               std::variant<std::string, py::object> const &type,
               std::vector<NodeContext> const &operands,
               ForgeOpAttrs const &named_attrs,
               std::vector<graphlib::OpType::Attr> const &attrs = {},
               bool copy_tms = true,
               bool dont_decompose = false,
               bool optimize_hoist = false,
               DataFormat output_df = DataFormat::Invalid)
            {
                if (std::holds_alternative<std::string>(type))
                {
                    TT_LOG_ASSERT(
                        not has_newstyle_interface(std::get<std::string>(type)),
                        "Error decomposing a type with old OpType interface, expects new OpType interface {}",
                        std::get<std::string>(type));
                    return self.op(
                        graphlib::OpType(std::get<std::string>(type), attrs, {}, named_attrs),
                        operands,
                        copy_tms,
                        dont_decompose,
                        optimize_hoist,
                        output_df);
                }
                else
                {
                    TT_ASSERT(attrs.size() == 0, "Illegal mixing of API modes");
                    auto const &op_type = std::get<py::object>(type).attr("op_type").cast<graphlib::OpType>();
                    return self.op(op_type, operands, copy_tms, dont_decompose, optimize_hoist, output_df);
                }
            },
            py::arg("type"),
            py::arg("operands"),
            py::arg("named_attrs"),
            py::arg("attrs") = std::vector<int>{},
            py::arg("copy_tms") = true,
            py::arg("dont_decompose") = false,
            py::arg("optimize_hoist") = false,
            py::arg("output_df") = DataFormat::Invalid)
        .def("fuse", &tt::DecomposingContext::fuse, py::arg("operand"), py::arg("producer_output_port_id") = 0)
        .def(
            "tensor",
            [](tt::DecomposingContext &self, py::object tensor)
            {
                return self.tensor(
                    make_shared_py_object(tensor),
                    graphlib::Shape::create(tensor.attr("shape").cast<std::vector<std::uint32_t>>()));
            },
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
            [](tt::DecomposingContext &self, graphlib::NodeContext const &node)
            {
                graphlib::Graph *graph = self.get_graph();
                std::vector<graphlib::Node *> operands = graph->data_operands(graph->node_by_id(node.id));
                std::vector<graphlib::NodeContext *> operand_contexts;
                for (auto it = operands.begin(); it != operands.end(); ++it)
                {
                    graphlib::NodeContext *op_context = new graphlib::NodeContext(*it);
                    operand_contexts.push_back(op_context);
                }
                return operand_contexts;
            })
        .def(
            "get_compiler_cfg",
            [](tt::DecomposingContext &self) { return borrow_shared_py_object(self.get_compiler_cfg()); });
}

}  // namespace tt
