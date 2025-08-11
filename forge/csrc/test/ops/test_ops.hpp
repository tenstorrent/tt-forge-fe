// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <ostream>
#include <unordered_map>
#include <utility>
#include <utils/env.hpp>

#include "autograd/autograd.hpp"
#include "graph_lib/defines.hpp"
#include "graph_lib/utils.hpp"
#include "passes/decomposing_context.hpp"
#include "test/common.hpp"

namespace tt::test::ops
{

using VecShapes = std::vector<graphlib::Shape>;

/// Simple struct to hold Op test parameters.
struct OpTestParam
{
    tt::ops::Op op;
    std::vector<graphlib::Shape> input_shapes;

    OpTestParam(const std::tuple<tt::ops::Op, std::vector<graphlib::Shape>>& param_tuple) :
        op{std::move(std::get<0>(param_tuple))}, input_shapes{std::move(std::get<1>(param_tuple))}
    {
    }

    OpTestParam(tt::ops::Op op, std::vector<graphlib::Shape> input_shapes) :
        op{std::move(op)}, input_shapes{std::move(input_shapes)}
    {
    }
};

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
    EXPECT_TRUE(torch::allclose(golden, output, /*rtol=*/1e-4, /*atol=*/1e-5))
        << "Tensors do not match: golden = " << golden << ", output = " << output;
}

/// Base class for Op tests.
///
/// For a given `Op` and input shapes, it creates a graph for it and provides methods to evaulate the graph,
/// run decompositions or autograd.
///
/// When the graph is initially created, we evaluate it once to get the golden tensors. Each further modification of
/// the graph (e.g., running autograd, decomposing the graph) can be checked by comparing the output tensors (after
/// graph modification) with the golden tensors.
class BaseOpTest : public ForgeGraphTest
{
   public:
    BaseOpTest(tt::ops::Op op, std::vector<graphlib::Shape> input_shapes) :
        op_{std::move(op)}, input_shapes_{std::move(input_shapes)}, input_tensors_{}, generated_grads_{}
    {
        torch::manual_seed(27);
    }

    BaseOpTest(const OpTestParam& param) : BaseOpTest(param.op, param.input_shapes) {}

    void SetUp() override
    {
        ForgeGraphTest::SetUp();

        // First forward pass (before any graph modification).
        // Save the results as golden tensors.
        auto output_tensors = eval_graph();
        golden_tensors_ = output_tensors;
    }

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

    /// Runs autograd on the graph and generates random tensors for gradient inputs.
    void run_autograd()
    {
        auto graph = get_graph();
        graph->set_training(true);

        EXPECT_TRUE(!graph->contains_bwd_nodes()) << "Graph contains backward nodes before running the autograd.";

        // Create the backward graph.
        autograd::autograd_config config{.recompute = false, .optimizer = py::none()};
        auto autograd_engine = tt::autograd::autograd_engine(graph, config);
        autograd_engine.run();

        run_decompose_graph<DecomposeEpoch::PostAutograd>();
        graph = get_graph();

        EXPECT_TRUE(graph->contains_bwd_nodes()) << "Graph does not contain backward nodes after running the autograd.";

        // For each gradient input node in the graph we need to generate a random tensor (so that we can run backward
        // evaluation afterwards).
        for (const auto& node : graph->nodes_by_type(graphlib::NodeType::kInput))
        {
            if (!node->as<graphlib::InputNode>()->is_gradient())
            {
                continue;
            }

            generated_grads_[node->name()] = torch::rand(torch_shape(node->shape()));
        }

        if (GraphTestFlagsEnvironment::instance()->enable_reportify)
        {
            graph->dump("post_autograd");
        }
    }

    /// Evaluates the graph node by node (for the provided epoch type).
    /// Returns a map of output tensor names to their corresponding torch::Tensor values.
    std::unordered_map<std::string, torch::Tensor> eval_graph(
        graphlib::NodeEpochType epoch_type = graphlib::NodeEpochType::Forward)
    {
        /// TODO(#2644): There is a method `eval_graph()` in `graphlib` python bindings, which should be used here.
        ///              However, that method is in really bad state and calling it here would be similar amount of work
        ///              as implementing this method from scratch. After #2644 is resolved, we should swap this
        ///              implementation with calling `eval_graph()` from `graphlib` python bindings.
        graphlib::Graph* graph = get_graph();

        std::unordered_map<std::string, torch::Tensor> intermediate_tensors{};
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
                else if (input_node->is_tensor())
                {
                    auto tensor = py::handle(static_cast<PyObject*>(input_node->tensor().get()));

                    auto torch_tensor = tensor.cast<torch::Tensor>();
                    intermediate_tensors[input->name()] = torch_tensor;
                }
                else
                {
                    throw std::runtime_error("Input node " + input->name() + " is a constant with unknown type");
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

        std::unordered_map<std::string, torch::Tensor> output_tensors;
        for (const Node* node : graphlib::topological_sort(*graph))
        {
            // For forward graph evaluation, skip nodes that are not in the forward epoch.
            // For backward graph evaluation, we need both forward (intermediates) and backward nodes.
            if (epoch_type == graphlib::NodeEpochType::Forward &&
                node->get_epoch_type() != graphlib::NodeEpochType::Forward)
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
                        << "Tensor for node " << operand->name() << " not found in intermediate_tensors map";
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

                output_tensors[node->name()] = intermediate_tensors.at(output->name());
            }
        }

        return output_tensors;
    }

    /// Compares the output tensors from the graph evaluation with the golden tensors from the initial graph evaluation.
    void compare_with_golden(const std::unordered_map<std::string, torch::Tensor>& output_tensors)
    {
        EXPECT_EQ(golden_tensors_.size(), output_tensors.size())
            << "Golden tensors and output tensors should have the same size";

        for (const auto& [output_name, golden_tensor] : golden_tensors_)
        {
            EXPECT_TRUE(output_tensors.find(output_name) != output_tensors.end())
                << "Output tensor for node " << output_name << " not found in output_tensors_ map";
            const auto& output_tensor = output_tensors.at(output_name);

            // Verify that the golden tensor matches the output tensor.
            assert_equal(golden_tensor, output_tensor);
        }
    }

    /// Verifies that the calculated gradients from the backward graph are the same as the gradients computed by torch.
    void verify_bwd_gradients(std::unordered_map<std::string, torch::Tensor>& computed_grads)
    {
        run_torch_backward();

        for (const auto& input_node : get_graph()->ordered_module_inputs())
        {
            if (input_node->as<graphlib::InputNode>()->is_gradient())
            {
                continue;
            }

            EXPECT_TRUE(input_tensors_.find(input_node->name()) != input_tensors_.end())
                << "Input tensor for node " << input_node->name() << " not found in input_tensors_ map";

            // Find the input's gradient node.
            auto grad_edge = get_graph()->edges(
                input_node,
                [](const graphlib::Edge& edge)
                { return edge.edge_type == graphlib::EdgeType::kAutogradInputToGradientOut; });
            EXPECT_TRUE(grad_edge.size() == 1)
                << "Input node " << input_node->name() << " should have exactly one gradient edge, but found "
                << grad_edge.size() << " edges";

            auto grad_node = get_graph()->node_by_id(grad_edge.front().consumer_node_id);

            EXPECT_TRUE(input_tensors_.find(input_node->name()) != input_tensors_.end())
                << "Input tensor for node " << input_node->name() << " not found in input_tensors_ map";
            auto input_tensor = input_tensors_.at(input_node->name());
            EXPECT_TRUE(computed_grads.find(grad_node->name()) != computed_grads.end())
                << "Gradient tensor for node " << grad_node->name() << " not found in output_tensors_ map";
            auto grad_tensor = computed_grads.at(grad_node->name());

            // Verify that the input tensor's gradient matches our calculated gradient tensor.
            assert_equal(input_tensor.grad(), grad_tensor);
        }
    }

    torch::Tensor generated_grad() const { return generated_grads_.begin()->second; }

    /// Runs torch backward on the golden output tensors.
    void run_torch_backward()
    {
        for (const auto& [name, tensor] : golden_tensors_)
        {
            EXPECT_TRUE(tensor.requires_grad())
                << "Golden tensor " << name << " does not require gradient, cannot run backward";

            // Find the gradient tensor (input to bwd) for this output tensor.
            auto edge = get_graph()->edges(
                get_graph()->get_node_by_name(name),
                [](const graphlib::Edge& edge) { return edge.edge_type == graphlib::EdgeType::kAutogradOutputToLoss; });
            EXPECT_TRUE(edge.size() == 1)
                << "Golden tensor " << name << " should have exactly one gradient edge, but found " << edge.size()
                << " edges";

            // The gradient tensor is the input to the backward graph.
            auto output_node = get_graph()->node_by_id(edge.front().consumer_node_id);
            auto grad_name = output_node->name();

            EXPECT_TRUE(generated_grads_.find(grad_name) != generated_grads_.end())
                << "Generated gradient for tensor " << name << " not found in generated_grads_ map";
            tensor.backward(generated_grads_.at(grad_name));
        }
    }

    template <DecomposeEpoch epoch = DecomposeEpoch::Initial>
    void run_decompose_graph()
    {
        // TODO: decomposing context needs `compiler_cfg`; passing nullptr for now...
        tt::decompose_tt_forge_graph<epoch>(get_graph(), std::shared_ptr<void>(nullptr, [](void*) {}));
    }

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
    std::unordered_map<std::string, torch::Tensor> golden_tensors_;
};

struct SimpleOpTest : public BaseOpTest, testing::WithParamInterface<OpTestParam>
{
   public:
    SimpleOpTest() : BaseOpTest(GetParam()) {}

    static std::string get_test_name(const testing::TestParamInfo<SimpleOpTest::ParamType>& info)
    {
        const auto& param = info.param;
        static std::unordered_map<std::string, size_t> op_name_id_map;

        auto op_name = param.op.as_string();
        return param.op.as_string() + std::to_string(op_name_id_map[op_name]++);
    }
};

// Test class for ops that don't support backward pass
struct SimpleOpDecomposeOnlyTest : public BaseOpTest, testing::WithParamInterface<OpTestParam>
{
   public:
    SimpleOpDecomposeOnlyTest() : BaseOpTest(GetParam()) {}

    static std::string get_test_name(const testing::TestParamInfo<SimpleOpDecomposeOnlyTest::ParamType>& info)
    {
        const auto& param = info.param;
        static std::unordered_map<std::string, size_t> op_name_id_map;

        auto op_name = param.op.as_string();
        return param.op.as_string() + std::to_string(op_name_id_map[op_name]++);
    }
};

inline std::vector<graphlib::Shape> shape_range(
    std::vector<uint32_t> min_shape, std::vector<uint32_t> max_shape, size_t idx = 0)
{
    const size_t N = min_shape.size();
    std::vector<graphlib::Shape> shapes;
    for (uint32_t val_dim = min_shape[idx]; val_dim <= max_shape[idx]; ++val_dim)
    {
        min_shape[idx] = val_dim;
        if (idx == N - 1)
        {
            std::vector<uint32_t> shape_vec(min_shape.begin(), min_shape.end());
            shapes.push_back(graphlib::Shape::create(shape_vec));
            continue;
        }

        auto new_shapes = shape_range(min_shape, max_shape, idx + 1);
        shapes.reserve(shapes.size() + new_shapes.size());
        shapes.insert(shapes.end(), new_shapes.begin(), new_shapes.end());
    }
    return shapes;
}

}  // namespace tt::test::ops
