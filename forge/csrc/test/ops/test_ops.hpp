// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <ostream>

#include "autograd/autograd.hpp"
#include "graph_lib/utils.hpp"
#include "test/common.hpp"
#include "verif/verif_ops.hpp"

namespace tt::test::ops
{

struct OpTestParam
{
    tt::ops::Op op;
    std::vector<graphlib::Shape> input_shapes;

    OpTestParam(const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& param_tuple) :
        op{std::move(std::get<0>(param_tuple))}, input_shapes{std::move(std::get<1>(param_tuple))}
    {
    }
};

struct EvalResult
{
    torch::Tensor golden;
    torch::Tensor output;
};

using eval_function_t = std::function<torch::Tensor(const std::vector<torch::Tensor>&)>;

inline std::ostream& operator<<(std::ostream& os, const torch::ArrayRef<long> shape)
{
    os << "[";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        os << shape[i];
        if (i < shape.size() - 1)
            os << ", ";
    }
    os << "]";
    return os;
}

inline void verify_shape(
    const graphlib::Graph* graph, const graphlib::PyOpNode* op_node, const torch::ArrayRef<long> expected_shape)
{
    // Verify that the shape of the output tensor matches the expected shape.
    auto shape = op_node->shape().as_vector();
    EXPECT_TRUE(std::equal(shape.begin(), shape.end(), expected_shape.begin(), expected_shape.end()))
        << "Shape mismatch for node " << op_node->name() << ": expected " << expected_shape << ", got "
        << op_node->shape();
}

inline void assert_equal(const torch::Tensor& golden, const torch::Tensor& output)
{
    // Compare the two tensors for equality.
    EXPECT_TRUE(golden.is_same_size(output))
        << "Tensors have different sizes: golden = " << golden.sizes() << ", output = " << output.sizes();
    EXPECT_TRUE(torch::allclose(golden, output))
        << "Tensors do not match: golden = " << golden << ", output = " << output;
}

class BaseOpTest : public ForgeGraphTest
{
   public:
    BaseOpTest(tt::ops::Op op, std::vector<graphlib::Shape> input_shapes) :
        op_{std::move(op)}, input_shapes_{std::move(input_shapes)}, input_tensors_{}, generated_grads_{}
    {
        torch::manual_seed(27);
    }

    BaseOpTest(const OpTestParam& param) : BaseOpTest(param.op, param.input_shapes) {}

    std::vector<OpType*> create_graph() override
    {
        std::vector<graphlib::Node*> inputs;
        for (const auto& shape : input_shapes_)
        {
            inputs.push_back(create_activation(shape));
            inputs.back()->as<graphlib::InputNode>()->set_requires_grad(true);

            torch::Tensor tensor = torch::rand(torch_shape(shape));
            tensor.set_requires_grad(true);
            input_tensors_[inputs.back()->name()] = tensor;
        }

        auto op = create_op(op_, inputs);
        return {op};
    }

    void run_autograd()
    {
        auto graph = get_graph();
        graph->set_training(true);

        EXPECT_TRUE(!graph->contains_bwd_nodes()) << "Graph contains backward nodes before running the autograd.";

        // Create the backward graph.
        autograd::autograd_config config{.recompute = false, .optimizer = py::none()};
        auto autograd_engine = tt::autograd::autograd_engine(graph, config);
        autograd_engine.run();

        graph = get_graph();

        EXPECT_TRUE(graph->contains_bwd_nodes()) << "Graph does not contain backward nodes after running the autograd.";

        for (const auto& node : graph->nodes_by_type(graphlib::NodeType::kInput))
        {
            if (node->get_epoch_type() != graphlib::NodeEpochType::Backward)
            {
                // Skip forward inputs.
                continue;
            }

            if (!node->as<graphlib::InputNode>()->is_gradient())
            {
                // Skip inputs that are not gradients.
                continue;
            }

            // For each input in the backward graph, we need to generate a random tensor.
            generated_grads_[node->name()] = torch::rand(torch_shape(node->shape()));
        }
    }

    torch::Tensor eval_graph(graphlib::NodeEpochType epoch_type = graphlib::NodeEpochType::Forward)
    {
        graphlib::Graph* graph = get_graph();

        std::unordered_map<std::string, torch::Tensor> intermediate_tensors{input_tensors_};
        for (const Node* input : graph->nodes_by_type(graphlib::NodeType::kInput))
        {
            auto input_node = input->as<graphlib::InputNode>();

            if (input_node->is_constant())
            {
                auto input_node = input->as<graphlib::ConstantInputNode>();
                if (input_node->is_single_value())
                {
                    intermediate_tensors[input->name()] =
                        input_node->constant_value() * torch::ones(torch_shape(input_node->shape()));
                }
                else
                {
                    throw std::runtime_error(
                        "Input node " + input->name() + " is a constant with multiple values, which is not supported.");
                }
            }
            else
            {
                switch (input->get_epoch_type())
                {
                    case graphlib::NodeEpochType::Forward:
                        EXPECT_TRUE(input_tensors_.find(input->name()) != input_tensors_.end())
                            << "Input tensor for node " << input->name() << " not found in input_tensors_ map";
                        intermediate_tensors[input->name()] = input_tensors_.at(input->name());
                        break;
                    case graphlib::NodeEpochType::Backward:
                        EXPECT_TRUE(generated_grads_.find(input->name()) != generated_grads_.end())
                            << "Gradient tensor for node " << input->name() << " not found in generated_grads_ map";
                        intermediate_tensors[input->name()] = generated_grads_.at(input->name());
                        break;
                    default:
                        throw std::runtime_error(
                            "Unsupported epoch type for input node: " +
                            std::to_string(static_cast<int>(input->get_epoch_type())));
                }
            }
        }

        EXPECT_TRUE(
            graph
                ->nodes([epoch_type](Node* n)
                        { return n->get_epoch_type() == epoch_type && n->node_type() == graphlib::NodeType::kOutput; })
                .size() == 1)
            << "Graph should have exactly one output, but found " << graph->get_ordered_output_names().size()
            << " outputs";

        for (const Node* node : graphlib::topological_sort(*graph))
        {
            // Skip nodes that are not in the specified epoch type.
            if (node->get_epoch_type() != epoch_type)
            {
                continue;
            }

            if (node->node_type() == graphlib::NodeType::kPyOp)
            {
                const auto* op_node = node->as<graphlib::PyOpNode>();

                std::vector<torch::Tensor> input_tensors_for_op;
                for (const auto& operand : graph->data_operands(node))
                {
                    EXPECT_TRUE(intermediate_tensors.find(operand->name()) != intermediate_tensors.end())
                        << "Input tensor for node " << operand->name() << "not found in intermediate_tensors map";
                    input_tensors_for_op.push_back(intermediate_tensors.at(operand->name()));
                }

                // Evaluate the operation using the op's eval method
                torch::Tensor output_tensor = op_node->op_type().eval(input_tensors_for_op);

                // Confirm that the `node->shape()` is properly calculated.
                verify_shape(graph, op_node, output_tensor.sizes());

                intermediate_tensors[node->name()] = output_tensor;
            }
            else if (node->node_type() == graphlib::NodeType::kOutput)
            {
                EXPECT_TRUE(graph->data_operands(node).size() == 1)
                    << "Output node should have exactly one input, but found " << graph->data_operands(node).size()
                    << " inputs";

                const Node* output = graph->data_operands(node).front();
                EXPECT_TRUE(intermediate_tensors.find(output->name()) != intermediate_tensors.end())
                    << "Output tensor for node " << output->name() << " not found in intermediate_tensors map";

                return intermediate_tensors.at(output->name());
            }
        }

        throw std::runtime_error("Graph evaluation did not produce an output tensor");
    }

    /// @brief Evaluates the whole graph and compares the output to a golden evaluation function.
    /// @param golden_eval A function that takes a vector of input tensors and returns the expected output tensor.
    /// @return void
    ///
    /// This function retrieves the graph, collects input tensors, and evaluates each operation in the graph
    /// using the `.eval()` mechanism implemented for each operation type.
    ///
    /// Compares the evaluated output of the whole graph to the output of the golden evaluation function.
    ///
    /// Also verifies that the shapes of each operation in the graph match to the shapes of the tensors calculated
    /// by the `.eval()` method.
    void eval_graph_and_compare(eval_function_t golden_eval) { auto input_tensors = get_input_tensors(); }

    std::vector<torch::Tensor> get_input_tensors() const
    {
        std::vector<torch::Tensor> tensors;
        tensors.reserve(input_tensors_.size());
        for (const auto& [name, tensor] : input_tensors_)
        {
            tensors.push_back(tensor);
        }
        return tensors;
    }

    torch::Tensor generated_grad() const { return generated_grads_.begin()->second; }

   private:
    // Convert our shape into torch shape.
    std::vector<long> torch_shape(const graphlib::Shape& shape)
    {
        std::vector<long> torch_shape;
        torch_shape.reserve(shape.size());
        for (const auto& dim : shape.as_vector())
        {
            torch_shape.push_back(static_cast<long>(dim));
        }
        return torch_shape;
    }

    tt::ops::Op op_;
    std::vector<graphlib::Shape> input_shapes_;
    std::unordered_map<std::string, torch::Tensor> input_tensors_;
    std::unordered_map<std::string, torch::Tensor> generated_grads_;
};

template <size_t N>
auto shape_range(const std::array<uint32_t, N>& start, const std::array<uint32_t, N>& end, uint32_t step = 1)
{
    static_assert(N <= 4, "Shape range can only be created for up to 4 dimensions.");
    if constexpr (N == 1)
    {
        return testing::ConvertGenerator(
            testing::Range(start[0], end[0], step),
            [](const uint32_t& param) { return std::vector{graphlib::Shape({param})}; });
    }
    else if constexpr (N == 2)
    {
        return testing::ConvertGenerator(
            testing::Combine(testing::Range(start[0], end[0], step), testing::Range(start[1], end[1], step)),
            [](const std::tuple<uint32_t, uint32_t>& params) { return std::vector{graphlib::Shape(params)}; });
    }
    else if constexpr (N == 3)
    {
        return testing::ConvertGenerator(
            testing::Combine(
                testing::Range(start[0], end[0], step),
                testing::Range(start[1], end[1], step),
                testing::Range(start[2], end[2], step)),
            [](const std::tuple<uint32_t, uint32_t, uint32_t>& params)
            { return std::vector{graphlib::Shape(params)}; });
    }
    else
    {
        return testing::ConvertGenerator(
            testing::Combine(
                testing::Range(start[0], end[0], step),
                testing::Range(start[1], end[1], step),
                testing::Range(start[2], end[2], step),
                testing::Range(start[3], end[3], step)),
            [](const std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>& params)
            { return std::vector{graphlib::Shape(params)}; });
    }
}

#define STANDARD_RANGE_OP_TEST_SET(op_name, op, test_class)                                                            \
    INSTANTIATE_TEST_SUITE_P(                                                                                          \
        op_name##Op1DRange,                                                                                            \
        test_class,                                                                                                    \
        testing::ConvertGenerator(                                                                                     \
            testing::Combine(                                                                                          \
                testing::Values(op), shape_range(std::array<uint32_t, 1>{1}, std::array<uint32_t, 1>{5})),             \
            [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }));              \
                                                                                                                       \
    INSTANTIATE_TEST_SUITE_P(                                                                                          \
        op_name##Op2DRange,                                                                                            \
        test_class,                                                                                                    \
        testing::ConvertGenerator(                                                                                     \
            testing::Combine(                                                                                          \
                testing::Values(op), shape_range(std::array<uint32_t, 2>{1, 1}, std::array<uint32_t, 2>{5, 5})),       \
            [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }));              \
                                                                                                                       \
    INSTANTIATE_TEST_SUITE_P(                                                                                          \
        op_name##Op3DRange,                                                                                            \
        test_class,                                                                                                    \
        testing::ConvertGenerator(                                                                                     \
            testing::Combine(                                                                                          \
                testing::Values(op), shape_range(std::array<uint32_t, 3>{1, 1, 1}, std::array<uint32_t, 3>{5, 1, 5})), \
            [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }));              \
                                                                                                                       \
    INSTANTIATE_TEST_SUITE_P(                                                                                          \
        op_name##Op4DRange,                                                                                            \
        test_class,                                                                                                    \
        testing::ConvertGenerator(                                                                                     \
            testing::Combine(                                                                                          \
                testing::Values(op),                                                                                   \
                shape_range(std::array<uint32_t, 4>{1, 1, 1, 1}, std::array<uint32_t, 4>{5, 1, 1, 5})),                \
            [](const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& params) { return params; }));

}  // namespace tt::test::ops
