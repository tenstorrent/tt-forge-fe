// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#include <pybind11/pybind11.h>
#pragma clang diagnostic pop

#include <pybind11/stl.h>
#include "pybind11_json/pybind11_json.hpp"


#include <sstream>
namespace py = pybind11;

#include "autograd/python_bindings.hpp"
#include "backend_api/device_config.hpp"
#include "forge_passes.hpp"
#include "forge_graph_module.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/python_bindings.hpp"
#include "lower_to_forge/common.hpp"
#include "passes/amp.hpp"
#include "passes/consteval.hpp"
#include "passes/fracture.hpp"
#include "passes/link_past_cache_ios.hpp"
#include "passes/move_index_to_mm_weights.hpp"
#include "passes/passes_utils.hpp"
#include "passes/python_bindings.hpp"
#include "passes/mlir_compiler.hpp"
#include "passes/split_graph.hpp"
#include "python_bindings_common.hpp"
#include "reportify/reportify.hpp"
#include "runtime/python_bindings.hpp"
#include "shared_utils/sparse_matmul_utils.hpp"
#include "tt_torch_device/python_bindings.hpp"
#include "utils/ordered_associative_containers/ordered_map.hpp"
#include "utils/signal_handlers.hpp"

namespace tt {

PYBIND11_MODULE(_C, m) {

    // Register signal handlers when loading forge module.
    static SignalHandlers signal_handlers;

    m.attr("__name__") = "forge._C";
    m.doc() = "python bindings to forge framwork";

    m.attr("VERSION") = py::int_(1);

    m.attr("k_dim") = py::int_(passes::k_dim);

    py::enum_<tt::ARCH>(m, "Arch")
        .value("JAWBRIDGE", tt::ARCH::JAWBRIDGE)
        .value("GRAYSKULL", tt::ARCH::GRAYSKULL)
        .value("WORMHOLE", tt::ARCH::WORMHOLE)
        .value("WORMHOLE_B0", tt::ARCH::WORMHOLE_B0)
        .value("BLACKHOLE", tt::ARCH::BLACKHOLE)
        .value("Invalid", tt::ARCH::Invalid)
        .export_values();

    py::enum_<tt::DataFormat>(m, "DataFormat")
        .value("Float32", tt::DataFormat::Float32)
        .value("Float16", tt::DataFormat::Float16)
        .value("Bfp8", tt::DataFormat::Bfp8)
        .value("Bfp4", tt::DataFormat::Bfp4)
        .value("Bfp2", tt::DataFormat::Bfp2)
        .value("Float16_b", tt::DataFormat::Float16_b)
        .value("Bfp8_b", tt::DataFormat::Bfp8_b)
        .value("Bfp4_b", tt::DataFormat::Bfp4_b)
        .value("Bfp2_b", tt::DataFormat::Bfp2_b)
        .value("Lf8", tt::DataFormat::Lf8)
        .value("UInt16", tt::DataFormat::UInt16)
        .value("Int8", tt::DataFormat::Int8)
        .value("RawUInt8", tt::DataFormat::RawUInt8)
        .value("RawUInt16", tt::DataFormat::RawUInt16)
        .value("RawUInt32", tt::DataFormat::RawUInt32)
        .value("Int32", tt::DataFormat::Int32)
        .value("Invalid", tt::DataFormat::Invalid)
        .export_values()
        .def(
            "to_json",
            [](tt::DataFormat df) {
                std::stringstream ss;
                ss << df;
                return ss.str();
            })
        .def("from_json", [](std::string const &encoded) {
            static std::unordered_map<std::string, tt::DataFormat> decode = {
                {"Float32", tt::DataFormat::Float32},
                {"Float16", tt::DataFormat::Float16},
                {"Bfp8", tt::DataFormat::Bfp8},
                {"Bfp4", tt::DataFormat::Bfp4},
                {"Bfp2", tt::DataFormat::Bfp2},
                {"Float16_b", tt::DataFormat::Float16_b},
                {"Bfp8_b", tt::DataFormat::Bfp8_b},
                {"Bfp4_b", tt::DataFormat::Bfp4_b},
                {"Bfp2_b", tt::DataFormat::Bfp2_b},
                {"Lf8", tt::DataFormat::Lf8},
                {"UInt16", tt::DataFormat::UInt16},
                {"Int8", tt::DataFormat::Int8},
                {"RawUInt8", tt::DataFormat::RawUInt8},
                {"RawUInt16", tt::DataFormat::RawUInt16},
                {"RawUInt32", tt::DataFormat::RawUInt32},
                {"Int32", tt::DataFormat::Int32},
                {"Invalid", tt::DataFormat::Invalid},
            };
            return decode.at(encoded);
        });

    py::module_ m_graph = m.def_submodule("graph", "Submodule defining forge graph functions");
    GraphModule(m_graph);

    py::enum_<tt::GraphType>(m, "GraphType")
        .value("Forward", tt::GraphType::Forward)
        .value("Backward", tt::GraphType::Backward)
        .value("Optimizer", tt::GraphType::Optimizer)
        .export_values();

    py::class_<tt::ForgeGraphModule>(m, "ForgeGraphModule")
        .def(py::init<std::string, tt::graphlib::Graph *>(), py::arg("name"), py::arg("forward_graph"))
        .def("set_graph", &tt::ForgeGraphModule::set_graph)
        .def("get_graph", &tt::ForgeGraphModule::get_graph);

    py::module_ m_autograd = m.def_submodule("autograd", "Submodule defining autograd_engine.");
    AutogradModule(m_autograd);

    py::module_ m_passes = m.def_submodule("passes", "API to Forge Passes");
    PassesModule(m_passes);

    py::module_ m_torch_device = m.def_submodule("torch_device", "TT Torch Device");
    TorchDeviceModule(m_torch_device);

    py::module m_runtime = m.def_submodule("runtime", "Submodule defining runtime functions");
    RuntimeModule(m_runtime);

    py::enum_<tt::MathFidelity>(m, "MathFidelity")
        .value("LoFi", tt::MathFidelity::LoFi)
        .value("HiFi2", tt::MathFidelity::HiFi2)
        .value("HiFi3", tt::MathFidelity::HiFi3)
        .value("HiFi4", tt::MathFidelity::HiFi4)
        .value("Invalid", tt::MathFidelity::Invalid)
        .export_values()
        .def(
            "to_json",
            [](tt::MathFidelity df)
            {
                std::stringstream ss;
                ss << df;
                return ss.str();
            })
        .def(
            "from_json",
            [](std::string const &encoded)
            {
                static std::unordered_map<std::string, tt::MathFidelity> decode = {
                    {"LoFi", tt::MathFidelity::LoFi},
                    {"HiFi2", tt::MathFidelity::HiFi2},
                    {"HiFi3", tt::MathFidelity::HiFi3},
                    {"HiFi4", tt::MathFidelity::HiFi4},
                    {"Invalid", tt::MathFidelity::Invalid},
                };
                return decode.at(encoded);
            });

    py::register_exception<UnsupportedHWOpsError>(m, "UnsupportedHWOpsError");

    m.def("link_past_cache_ios", &passes::link_past_cache_ios);
    m.def("move_index_to_mm_weights", &passes::move_index_to_mm_weights);
    m.def("run_post_initial_graph_passes", &run_post_initial_graph_passes);
    m.def("run_optimization_graph_passes", &run_optimization_graph_passes);
    m.def("run_post_optimize_decompose_graph_passes", &run_post_optimize_decompose_graph_passes);
    m.def("run_consteval_graph_pass", &passes::run_consteval_graph_pass);
    m.def("run_post_autograd_graph_passes", &run_post_autograd_graph_passes);
    m.def(
        "run_pre_placer_forge_passes",
        &run_pre_placer_forge_passes,
        py::arg("graph"),
        py::arg("device_config"),
        py::arg("chip_ids") = std::vector<std::uint32_t>{0},
        py::arg("op_names_dont_fuse") = std::vector<std::string>{},
        py::arg("op_names_manual_fuse") = std::vector<std::string>{},
        py::arg("fracture_chip_id_assignments") = passes::FractureChipIdAssignments{},
        py::arg("default_df_override") = std::optional<DataFormat>{},
        py::arg("default_accumulate_df") = std::optional<DataFormat>{},
        py::arg("enable_broadcast_splitting") = false,
        py::arg("fp32_fallback") = DataFormat::Float16_b,
        py::arg("default_math_fidelity") = MathFidelity::HiFi3,
        py::arg("enable_auto_fusing") = false,
        py::arg("amp_level") = 0,
        py::arg("enable_recompute") = 0,
        py::arg("output_queues_on_host") = true,
        py::arg("input_queues_on_host") = true,
        py::arg("insert_queues") = std::vector<std::tuple<std::string, std::string, int>>{},
        py::arg("amp_properties") = std::vector<AMPNodeProperties>{},
        py::arg("op_intermediates_to_save") = std::vector<std::string>{},
        py::arg("use_interactive_placer") = true,
        py::arg("enable_device_tilize") = false);
    m.def(
        "run_pre_lowering_passes",
        &run_pre_lowering_passes,
        py::arg("graph"),
        py::arg("default_df_override") = std::optional<DataFormat>{});
    m.def("run_mlir_compiler", &passes::run_mlir_compiler);
    m.def("split_graph", &passes::split_graph);

    m.def(
        "dump_graph",
        [](const tt::graphlib::Graph *graph, std::string test_name, std::string graph_name)
        { tt::reportify::dump_graph(test_name, graph_name, graph); },
        py::arg("graph"),
        py::arg("test_name"),
        py::arg("graph_name"));
    m.def(
        "dump_epoch_type_graphs",
        [](const tt::graphlib::Graph *graph, std::string test_name, std::string graph_name)
        { tt::reportify::dump_epoch_type_graphs(test_name, graph_name, graph); },
        py::arg("graph"),
        py::arg("test_name"),
        py::arg("graph_name"));
    m.def(
        "dump_epoch_id_graphs",
        [](const tt::graphlib::Graph *graph, std::string test_name, std::string graph_name)
        { tt::reportify::dump_epoch_id_graphs(test_name, graph_name, graph); },
        py::arg("graph"),
        py::arg("test_name"),
        py::arg("graph_name"));

    py::enum_<tt::graphlib::NodeEpochType>(m, "NodeEpochType")
        .value("Forward", tt::graphlib::NodeEpochType::Forward)
        .value("Backward", tt::graphlib::NodeEpochType::Backward)
        .value("Optimizer", tt::graphlib::NodeEpochType::Optimizer)
        .export_values();

    py::class_<tt::sparse::SparseCOO>(m, "SparseCOO")
        .def(
            py::init<
                std::vector<std::int64_t>,
                std::vector<std::int64_t>,
                std::vector<float>,
                std::vector<std::int64_t>>(),
            py::arg("rows"),
            py::arg("cols"),
            py::arg("vals"),
            py::arg("shape"))
        .def_readonly("shape", &sparse::SparseCOO::shape)
        .def_readonly("rows", &sparse::SparseCOO::rows)
        .def_readonly("cols", &sparse::SparseCOO::cols)
        .def_readonly("vals", &sparse::SparseCOO::vals);

    py::class_<tt::sparse::SparseFORGE>(m, "SparseFORGE")
        .def_readonly("sparse_indices", &sparse::SparseFORGE::sparse_indices)
        .def_readonly("sparse_shape", &sparse::SparseFORGE::sparse_shape)
        .def_readonly("zdim", &sparse::SparseFORGE::zdim)
        .def_readonly("bcast_factor", &sparse::SparseFORGE::bcast_factor)
        .def("get_sparse_tile_ptr_bits", &sparse::SparseFORGE::get_sparse_tile_ptr_bits)
        .def("get_sparse_ublock_idx_bits", &sparse::SparseFORGE::get_sparse_ublock_idx_bits)
        .def("get_sparse_tiles_and_encodings", [](tt::sparse::SparseFORGE &self, int grid_r) {
            return self.get_sparse_tiles_and_encodings(grid_r);
        });

    // m.def("compress_sparse_tensor", &sparse::compress_sparse_tensor);
    m.def("compress_sparse_tensor_and_strip_info", &sparse::compress_sparse_tensor_and_strip_info);

    py::class_<tt::passes::AMPNodeProperties>(m, "AMPNodeProperties")
        .def(
            py::init<
                std::optional<std::string>,
                std::optional<tt::graphlib::NodeEpochType>,
                std::optional<tt::DataFormat>,
                std::optional<tt::DataFormat>,
                std::optional<tt::DataFormat>,
                std::optional<tt::MathFidelity>,
                std::optional<std::string>,
                std::optional<tt::passes::InputDfConfig>,
                std::optional<bool>,
                std::optional<std::vector<std::pair<std::uint32_t, std::uint32_t>>>>(),
            py::arg("op_type") = std::nullopt,
            py::arg("epoch_type") = std::nullopt,
            py::arg("output_df") = std::nullopt,
            py::arg("intermediate_df") = std::nullopt,
            py::arg("accumulate_df") = std::nullopt,
            py::arg("math_fidelity") = std::nullopt,
            py::arg("name_regex_match") = std::nullopt,
            py::arg("input_df") = std::nullopt,
            py::arg("is_gradient_op") = std::nullopt,
            py::arg("input_parameter_indices_to_optimize") = std::nullopt)
        .def_readonly("op_type", &tt::passes::AMPNodeProperties::op_type)
        .def_readonly("epoch_type", &tt::passes::AMPNodeProperties::epoch_type)
        .def_readonly("output_df", &tt::passes::AMPNodeProperties::output_df)
        .def_readonly("intermediate_df", &tt::passes::AMPNodeProperties::intermediate_df)
        .def_readonly("accumulate_df", &tt::passes::AMPNodeProperties::accumulate_df)
        .def_readonly("math_fidelity", &tt::passes::AMPNodeProperties::math_fidelity)
        .def_readonly("input_df", &tt::passes::AMPNodeProperties::input_df)
        .def_readonly("is_gradient_op", &tt::passes::AMPNodeProperties::is_gradient_op)
        .def_readonly("name_regex_match", &tt::passes::AMPNodeProperties::name_regex_match)
        .def_readonly(
            "input_parameter_indices_to_optimize", &tt::passes::AMPNodeProperties::input_parameter_indices_to_optimize)
        .def("to_json", [](tt::passes::AMPNodeProperties const& properties) {
            nlohmann::json j = properties;
            return j;
        })
        .def("from_json", [](nlohmann::json const& j) {
            return j.template get<tt::passes::AMPNodeProperties>();
        })
        .def(py::pickle(
            [](const tt::passes::AMPNodeProperties &p) {  // __getstate__
                return py::make_tuple(
                    p.op_type,
                    p.epoch_type,
                    p.output_df,
                    p.intermediate_df,
                    p.accumulate_df,
                    p.math_fidelity,
                    p.name_regex_match,
                    p.input_df,
                    p.is_gradient_op,
                    p.input_parameter_indices_to_optimize);
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 10)
                    throw std::runtime_error("Invalid state!");

                tt::passes::AMPNodeProperties p(
                    t[0].cast<std::optional<std::string>>(),
                    t[1].cast<std::optional<tt::graphlib::NodeEpochType>>(),
                    t[2].cast<std::optional<tt::DataFormat>>(),
                    t[3].cast<std::optional<tt::DataFormat>>(),
                    t[4].cast<std::optional<tt::DataFormat>>(),
                    t[5].cast<std::optional<tt::MathFidelity>>(),
                    t[6].cast<std::optional<std::string>>(),
                    t[7].cast<std::optional<tt::passes::InputDfConfig>>(),
                    t[8].cast<std::optional<bool>>(),
                    t[9].cast<std::optional<std::vector<std::pair<std::uint32_t, std::uint32_t>>>>());
                return p;
            }));
}

}
