// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/amp.hpp"

#include <filesystem>
#include <fstream>
#include <functional>
#include <regex>
#include <sstream>

#include "backend_api/device_config.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "lower_to_forge/common.hpp"
#include "nlohmann/json.hpp"
#include "passes/dataformat.hpp"
#include "reportify/paths.hpp"
#include "reportify/to_json.hpp"
#include "utils/logger.hpp"

using Graph = tt::graphlib::Graph;
using Node = tt::graphlib::Node;

namespace impl
{

template <typename Predicate>
auto conjunction(Predicate&& predicate)
{
    return [predicate](auto&& arg) { return predicate(std::forward<decltype(arg)>(arg)); };
}

template <typename Predicate, typename... RemainingPredicates>
auto conjunction(Predicate&& predicate, RemainingPredicates&&... remaining_predicates)
{
    return [predicate, remaining_predicates...](auto&& arg) -> bool
    {
        return predicate(std::forward<decltype(arg)>(arg)) and
               conjunction(remaining_predicates...)(std::forward<decltype(arg)>(arg));
    };
}

template <typename UnaryPredicate>
std::vector<Node*> get_queried_nodes(const Graph* graph, UnaryPredicate&& unary_predicate)
{
    const std::vector<Node*>& ops = graph->nodes();
    std::vector<Node*> filtered_ops;
    std::copy_if(ops.begin(), ops.end(), std::back_inserter(filtered_ops), unary_predicate);
    return filtered_ops;
}

}  // namespace impl

namespace tt::passes
{

using impl::conjunction;
using impl::get_queried_nodes;

bool RegexMatcher::has_matching_string(const std::string& regex_string, const std::string& candidate_string)
{
    // Immediately return true if regex_string is empty
    if (regex_string.empty())
    {
        return true;
    }
    std::smatch base_match;
    // Check if the regex is already compiled in the cache
    auto it = regex_cache.find(regex_string);
    if (it == regex_cache.end())
    {
        // Compile the regex and store it in the cache
        std::regex compiled_regex(regex_string);
        regex_cache[regex_string] = compiled_regex;
        return std::regex_match(candidate_string, base_match, compiled_regex);
    }
    else
    {
        // Use the compiled regex from the cache
        return std::regex_match(candidate_string, base_match, it->second);
    }
}

template <class T>
std::string get_string(T obj)
{
    std::ostringstream oss;
    oss << obj;
    return oss.str();
}

void to_json(nlohmann::json& j, const AMPNodeProperties& p)
{
    j = nlohmann::json{
        {"op_type", p.op_type},
        {"epoch_type", p.epoch_type},
        {"output_df", p.output_df},
        {"intermediate_df", p.intermediate_df},
        {"accumulate_df", p.accumulate_df},
        {"math_fidelity", p.math_fidelity},
        {"name_regex_match", p.name_regex_match},
        {"input_df", p.input_df},
        {"is_gradient_op", p.is_gradient_op},
        {"input_parameter_indices_to_optimize", p.input_parameter_indices_to_optimize}};
}

void from_json(const nlohmann::json& j, AMPNodeProperties& p)
{
    j.at("op_type").get_to(p.op_type);
    j.at("epoch_type").get_to(p.epoch_type);
    j.at("output_df").get_to(p.output_df);
    j.at("intermediate_df").get_to(p.intermediate_df);
    j.at("accumulate_df").get_to(p.accumulate_df);
    j.at("math_fidelity").get_to(p.math_fidelity);
    j.at("name_regex_match").get_to(p.name_regex_match);
    j.at("input_df").get_to(p.input_df);
    j.at("is_gradient_op").get_to(p.is_gradient_op);
    j.at("input_parameter_indices_to_optimize").get_to(p.input_parameter_indices_to_optimize);
}

DataFormat get_reduced_precision_format(const DataFormat& from_df, std::uint32_t target_mantissa_bits)
{
    const std::unordered_map<std::uint32_t, DataFormat> bits_to_a_df = {
        {8, DataFormat::Bfp8},
        {4, DataFormat::Bfp4},
        {2, DataFormat::Bfp2},
    };
    const std::unordered_map<std::uint32_t, DataFormat> bits_to_b_df = {
        {8, DataFormat::Bfp8_b},
        {4, DataFormat::Bfp4_b},
        {2, DataFormat::Bfp2_b},
    };
    if (bits_to_a_df.find(target_mantissa_bits) == bits_to_a_df.end())
    {
        log_fatal("User has defined reduced mantissa bits to be an invalid configuration: {}", target_mantissa_bits);
    }
    return is_a_data_format(from_df) ? bits_to_a_df.at(target_mantissa_bits) : bits_to_b_df.at(target_mantissa_bits);
}

void configure_node_from_properties(const Graph* graph, Node* node, const AMPNodeProperties& properties)
{
    if (properties.output_df.has_value())
    {
        log_debug(
            LogGraphCompiler,
            "\t{} setting output_df from {} to {}",
            node->name(),
            node->output_df(),
            properties.output_df.value());
        node->set_output_df(properties.output_df.value());
    }
}

void apply_optimization(const Graph* graph, const std::vector<Node*>& nodes, const AMPNodeProperties& properties)
{
    for (Node* node : nodes)
    {
        log_debug(LogGraphCompiler, "{} is matched from user configuration", node->name());
        configure_node_from_properties(graph, node, properties);
    }
}

std::unordered_map<std::string, AMPNodeProperties> get_node_to_amp_properties(Graph* graph)
{
    std::unordered_map<std::string, AMPNodeProperties> node_to_amp_properties;

    for (Node* node : graph->nodes())
    {
        if (node->node_type() == graphlib::NodeType::kInput or node->node_type() == graphlib::NodeType::kOutput or
            node->node_type() == graphlib::NodeType::kQueue)
        {
            node_to_amp_properties[node->name()] =
                AMPNodeProperties("queue", node->get_epoch_type(), node->output_df());
        }
    }
    return node_to_amp_properties;
}

nlohmann::json get_node_to_amp_properties_json(Graph* graph)
{
    nlohmann::json output_json;
    auto properties = get_node_to_amp_properties(graph);
    for (const auto& [node_name, amp_properties] : properties)
    {
        output_json[node_name] = nlohmann::json(amp_properties);
    }

    for (Node* node : graph->nodes())
    {
        output_json[node->name()]["tags"] = node->as<graphlib::TaggedNode>()->get_tags();
    }
    return output_json;
}

void dump_mixed_precision_json_to_file(graphlib::Graph* graph, std::optional<std::string> filepath)
{
    if (env_as<bool>("FORGE_DISABLE_REPORTIFY_DUMP"))
        return;
    bool enable_dump = env_as<bool>("FORGE_DUMP_MIXED_PRECISION");

    if (not enable_dump)
    {
        return;
    }

    std::string filename;
    if (not filepath.has_value())
    {
        std::string output_dir = reportify::get_default_reportify_path(graph->name());
        std::filesystem::create_directories(output_dir);
        filename = output_dir + "/amp_settings.json";
    }
    else
    {
        filename = filepath.value();
    }

    std::ofstream out(filename);
    TT_ASSERT(out.is_open(), "Can't open " + filename + " for writing.");
    nlohmann::json content = get_node_to_amp_properties_json(graph);
    out << content.dump(4);
    out.close();
}

void apply_O0_optimization(const Graph* graph)
{
    // Current default for O0 won't explicitly configure any ops.
    (void)graph;
}

std::optional<std::string> original_op_type(const graphlib::Node* node)
{
    if (not node->as<graphlib::TaggedNode>()->has_tag("original_op_type"))
    {
        return {};
    }
    graphlib::TagValue tag_value = node->as<graphlib::TaggedNode>()->tag_value("original_op_type");
    if (const std::string* pval = std::get_if<std::string>(&tag_value); pval != nullptr)
    {
        return *pval;
    }
    return {};
}

bool is_matched_op(const AMPNodeProperties& amp_properties, RegexMatcher& regex_matcher, const Node* node)
{
    bool is_match = true;
    if (amp_properties.name_regex_match.has_value())
    {
        is_match &= regex_matcher.has_matching_string(amp_properties.name_regex_match.value(), node->name());
    }
    if (amp_properties.epoch_type.has_value())
    {
        is_match &= amp_properties.epoch_type.value() == node->get_epoch_type();
    }
    if (amp_properties.is_gradient_op.has_value())
    {
        const graphlib::OpNode* op = dynamic_cast<const graphlib::OpNode*>(node);
        if (op != nullptr)
        {
            is_match &= amp_properties.is_gradient_op.value() == op->is_gradient_op();
        }
    }
    if (amp_properties.op_type.has_value())
    {
        const graphlib::OpNode* op_node = dynamic_cast<const graphlib::OpNode*>(node);
        if (op_node != nullptr)
        {
            is_match &=
                (amp_properties.op_type.value() == op_node->op_name() or
                 amp_properties.op_type.value() == original_op_type(node));
        }
        else if (auto input_node = dynamic_cast<const graphlib::InputNode*>(node); input_node != nullptr)
        {
            is_match &= amp_properties.op_type.value() == graphlib::to_string(input_node->input_type());
        }
        else
        {
            is_match &= false;
        }
    }
    return is_match;
};

void apply_configuration(const Graph* graph, const std::vector<AMPNodeProperties>& user_properties)
{
    RegexMatcher regex_matcher;
    for (const auto& amp_properties : user_properties)
    {
        auto is_matched_op_ = std::bind(is_matched_op, amp_properties, regex_matcher, std::placeholders::_1);

        apply_optimization(graph, get_queried_nodes(graph, is_matched_op_), amp_properties);
    }
}

// TODO(jchu): clean this up,
struct AMPNodePropertiesInternal
{
    std::optional<std::string> op_type = std::nullopt;
    std::optional<NodeEpochType> epoch_type = std::nullopt;
    std::optional<DataFormat> output_df = std::nullopt;
    std::optional<DataFormat> intermediate_df = std::nullopt;
    std::optional<DataFormat> accumulate_df = std::nullopt;
    std::optional<MathFidelity> math_fidelity = std::nullopt;
    std::optional<std::string> name_regex_match = std::nullopt;
    std::optional<InputDfConfig> input_df = std::nullopt;
    std::optional<bool> is_gradient_op = std::nullopt;
    std::optional<std::vector<std::pair<std::uint32_t, std::uint32_t>>> input_parameter_indices_to_optimize =
        std::nullopt;

    AMPNodeProperties create() const
    {
        return AMPNodeProperties(
            this->op_type,
            this->epoch_type,
            this->output_df,
            this->intermediate_df,
            this->accumulate_df,
            this->math_fidelity,
            this->name_regex_match,
            this->input_df,
            this->is_gradient_op,
            this->input_parameter_indices_to_optimize);
    }
};

void apply_mixed_b_optimization(const Graph* graph)
{
    log_debug(LogGraphCompiler, "Running with MixedB Precision");

    // Set LayerNorm and Softmax to FP16_B Format
    AMPNodePropertiesInternal softmax_config = {
        .op_type = "softmax",
        .output_df = DataFormat::Float16_b,
        .intermediate_df = DataFormat::Float16_b,
        .accumulate_df = DataFormat::Float16_b,
    };
    AMPNodePropertiesInternal layernorm_config = {
        .op_type = "layernorm",
        .output_df = DataFormat::Float16_b,
        .intermediate_df = DataFormat::Float16_b,
        .accumulate_df = DataFormat::Float16_b,
    };
    AMPNodePropertiesInternal matmul_config = {
        .op_type = "matmul",
        .output_df = DataFormat::Bfp8_b,
        .intermediate_df = DataFormat::Bfp8_b,
        .accumulate_df = DataFormat::Float16_b,
        .math_fidelity = MathFidelity::HiFi2,
        .input_df = InputIndexToConfig{{1, {DataFormat::Bfp8_b, false}}, {2, {DataFormat::Bfp8_b, false}}}};

    std::vector<AMPNodeProperties> default_opt_configuration = {
        softmax_config.create(),
        layernorm_config.create(),
        matmul_config.create(),
    };

    apply_configuration(graph, default_opt_configuration);
}

void apply_mixed_a_optimization(const Graph* graph)
{
    log_debug(LogGraphCompiler, "Running with MixedA Precision");

    auto default_matmul_config = AMPNodePropertiesInternal{
        .op_type = "matmul",
        .output_df = DataFormat::Bfp8,
        .intermediate_df = DataFormat::Float16,
        .accumulate_df = DataFormat::Float16,
        .input_df = InputIndexToConfig{
            {0, {DataFormat::Bfp8, true}}, {1, {DataFormat::Bfp8, true}}, {2, {DataFormat::Bfp8, true}}}};
    auto broadcast_matmul_config = AMPNodePropertiesInternal{
        .op_type = "matmul",
        .output_df = DataFormat::Float16,
        .intermediate_df = DataFormat::Float16,
        .accumulate_df = DataFormat::Float16,
        .name_regex_match = ".*brcst.*",
        .input_df = InputIndexToConfig{
            {0, {DataFormat::Float16, true}}, {1, {DataFormat::Float16, true}}, {2, {DataFormat::Float16, true}}}};
    auto softmax_matmul_config = AMPNodePropertiesInternal{
        .op_type = "matmul",
        .output_df = DataFormat::Float16,
        .intermediate_df = DataFormat::Float16,
        .accumulate_df = DataFormat::Float16,
        .name_regex_match = ".*softmax.*",
    };
    auto layernorm_matmul_config = AMPNodePropertiesInternal{
        .op_type = "matmul",
        .output_df = DataFormat::Float16,
        .intermediate_df = DataFormat::Float16,
        .accumulate_df = DataFormat::Float16,
        .name_regex_match = ".*layernorm.*"};
    auto softmax_multiply_config = AMPNodePropertiesInternal{
        .op_type = "multiply",
        .output_df = DataFormat::Float16,
        .intermediate_df = DataFormat::Float16,
        .accumulate_df = DataFormat::Float16,
        .name_regex_match = ".*softmax.*"};
    auto gelu_config = AMPNodePropertiesInternal{.op_type = "gelu", .intermediate_df = DataFormat::Bfp8};
    auto fused_op_config = AMPNodePropertiesInternal{
        .op_type = "fused_op",
        .output_df = DataFormat::Float16,
        .intermediate_df = DataFormat::Float16,
        .accumulate_df = DataFormat::Float16,
        .input_df = InputIndexToConfig{
            {0, {DataFormat::Float16, true}}, {1, {DataFormat::Float16, true}}, {2, {DataFormat::Float16, true}}}};
    auto buffer_config = AMPNodePropertiesInternal{
        .op_type = "buffer",
        .output_df = DataFormat::Float16,
        .intermediate_df = DataFormat::Float16,
        .accumulate_df = DataFormat::Float16};

    std::vector<AMPNodeProperties> default_opt_configuration = {
        default_matmul_config.create(),
        broadcast_matmul_config.create(),
        softmax_matmul_config.create(),
        layernorm_matmul_config.create(),
        softmax_multiply_config.create(),
        gelu_config.create(),
        fused_op_config.create(),
        buffer_config.create(),
    };

    apply_configuration(graph, default_opt_configuration);
}

enum class MixedPrecisionSetting
{
    None = 0,
    Mixed_B_Formats = 1,
    Mixed_A_Formats = 2,
};
using OptToFunctionMapping = std::unordered_map<MixedPrecisionSetting, std::function<void(Graph*)>>;
const OptToFunctionMapping opt_dispatch_table = {
    {MixedPrecisionSetting::None, apply_O0_optimization},                  // mixed a-formats; bert-large model
    {MixedPrecisionSetting::Mixed_A_Formats, apply_mixed_a_optimization},  // mixed a-formats; bert-large model
    {MixedPrecisionSetting::Mixed_B_Formats, apply_mixed_b_optimization},  // mixed b-formats; nlp models
};

void const_tag_propagation(Graph* graph)
{
    auto is_constant = [](const Node* node) -> bool
    { return (node->node_type() == graphlib::NodeType::kInput) and (node->as<graphlib::InputNode>()->is_constant()); };
    for (auto input_node : get_queried_nodes(graph, is_constant))
    {
        auto data_users = graph->data_users(input_node);
        auto ref_node = data_users.at(0);

        // user the user node to transfer the tags
        input_node->as<graphlib::TaggedNode>()->add_tags(ref_node->as<graphlib::TaggedNode>()->get_tags());
    }
}

static bool is_valid_opt_level(const int opt_level)
{
    return opt_level >= static_cast<int>(MixedPrecisionSetting::None) and
           opt_level <= static_cast<int>(MixedPrecisionSetting::Mixed_A_Formats);
}

MixedPrecisionSetting get_mixed_precision_settings(const std::optional<DataFormat> default_df_override, int opt_level)
{
    if (not is_valid_opt_level(opt_level))
    {
        log_warning(LogGraphCompiler, "User specified invalid AMP optimization level. Skipping AMP configuration.");
        opt_level = 0;
    }

    if (opt_level > 0 and default_df_override.has_value())
    {
        return is_b_data_format(default_df_override.value()) ? MixedPrecisionSetting::Mixed_B_Formats
                                                             : MixedPrecisionSetting::Mixed_A_Formats;
    }
    else
    {
        return static_cast<MixedPrecisionSetting>(opt_level);
    }
}

void run_automatic_mixed_precision(
    Graph* graph,
    const DeviceConfig& device_config,
    const std::optional<DataFormat> default_df_override,
    const int opt_level,
    const std::vector<AMPNodeProperties>& user_properties)
{
    log_info(LogGraphCompiler, "Running with Automatic Mixed Precision Level = {}.", opt_level);
    MixedPrecisionSetting setting = get_mixed_precision_settings(default_df_override, opt_level);
    opt_dispatch_table.at(setting)(graph);

    apply_configuration(graph, user_properties);

    // Revalidate; Fix illegal situations; automatic propagation
    satisfy_data_format_constraints(graph, device_config.supports_fp32_accumulation());

    const_tag_propagation(graph);

    dump_mixed_precision_json_to_file(graph);
}

}  // namespace tt::passes
