// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/pre_lowering_passes.hpp"
#include "test/common.hpp"

namespace tt::test
{
struct MMFuseBias
    : public ForgeGraphTest,
      public testing::WithParamInterface<std::tuple<int, int, std::vector<graphlib::OpType>, graphlib::NodeType>>
{
   protected:
    virtual std::vector<OpType *> create_graph() override
    {
        auto [param_c, bias_c, tms_between, expected_node_type] = GetParam();

        int seq_len = 32;
        int hidden = 128;

        auto act = create_activation(shape(1, 1, seq_len, hidden));
        auto weight = create_parameter(shape(1, 1, hidden, param_c));
        auto bias = create_parameter(shape(1, 1, 1, bias_c));

        auto matmul = create_op("matmul", {act, weight});
        matmul_op_name = matmul->name();

        for (auto op : tms_between)
        {
            matmul = create_op(op, {matmul});
        }

        auto add = create_op("add", {matmul, bias});

        return {add};
    }

    std::string matmul_op_name;
};

INSTANTIATE_TEST_SUITE_P(
    MMFuseBias,
    MMFuseBias,
    testing::Values(
        std::make_tuple(64, 64, std::vector<graphlib::OpType>{}, graphlib::NodeType::kOutput),
        std::make_tuple(32, 32, std::vector<graphlib::OpType>{}, graphlib::NodeType::kOutput),
        std::make_tuple(128, 1, std::vector<graphlib::OpType>{}, graphlib::NodeType::kOutput),
        std::make_tuple(
            64,
            64,
            std::vector<graphlib::OpType>{
                graphlib::OpType("squeeze", {0}),
                graphlib::OpType("unsqueeze", {0, 3}),
            },
            graphlib::NodeType::kPyOp)));

}  // namespace tt::test
