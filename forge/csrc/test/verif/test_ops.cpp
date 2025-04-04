// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "test/common.hpp"
#include "verif/verif_ops.hpp"

namespace tt::test
{

class VerifTest : public ::testing::Test, public ::testing::WithParamInterface<at::ScalarType>
{
   public:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_P(VerifTest, test_all_close)
{
    auto dtype = GetParam();
    torch::Tensor a = torch::tensor({1.0, 2.0, 3.0}, at::TensorOptions().dtype(dtype));
    torch::Tensor b = torch::tensor({1.0, 2.0, 3.0}, at::TensorOptions().dtype(dtype));
    EXPECT_TRUE(tt::all_close(a, b));

    a = torch::tensor({1.0, 2.0, 3.0}, at::TensorOptions().dtype(dtype));
    b = torch::tensor({0.0, 2.0, 3.0}, at::TensorOptions().dtype(dtype));
    EXPECT_FALSE(tt::all_close(a, b));
}

TEST_P(VerifTest, test_pcc)
{
    auto dtype = GetParam();

    torch::Tensor a = torch::tensor({1.0, 2.0, 3.0}, at::TensorOptions().dtype(dtype));
    torch::Tensor b = torch::tensor({1.0, 2.0, 3.0}, at::TensorOptions().dtype(dtype));
    EXPECT_TRUE(tt::calculate_tensor_pcc(a, b) > 0.99);

    a = torch::tensor({1.0, 2.0, 3.0}, at::TensorOptions().dtype(dtype));
    b = torch::tensor({0.0, 2.0, 3.0}, at::TensorOptions().dtype(dtype));
    std::cerr << "pcc: " << tt::calculate_tensor_pcc(a, b) << std::endl;
    EXPECT_TRUE(tt::calculate_tensor_pcc(a, b) < 0.99);
}

TEST_P(VerifTest, test_max_abs_diff)
{
    auto dtype = GetParam();

    torch::Tensor a = torch::tensor({1.0, 2.0, 3.0}, at::TensorOptions().dtype(dtype));
    torch::Tensor b = torch::tensor({1.0, 2.0, 3.0}, at::TensorOptions().dtype(dtype));
    EXPECT_TRUE(tt::max_abs_diff(a, b) == 0);

    a = torch::tensor({1.0, 2.0, 3.0}, at::TensorOptions().dtype(dtype));
    b = torch::tensor({0.0, 2.0, 3.0}, at::TensorOptions().dtype(dtype));
    EXPECT_TRUE(tt::max_abs_diff(a, b) == 1);
}

INSTANTIATE_TEST_SUITE_P(
    VerifTest,
    VerifTest,
    ::testing::Values(at::kFloat, at::kBFloat16, at::kHalf, at::kDouble, at::kInt, at::kLong, at::kShort));
// ::testing::PrintToStringParamName());

}  // namespace tt::test
