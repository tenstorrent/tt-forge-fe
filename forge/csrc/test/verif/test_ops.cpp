// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>

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

TEST_F(VerifTest, test_pcc_large)
{
    torch::manual_seed(46);
    torch::Tensor a = torch::rand({1000, 1000}, at::TensorOptions().dtype(at::kFloat));
    torch::Tensor b = torch::rand({1000, 1000}, at::TensorOptions().dtype(at::kFloat));

    auto calculated_pcc = tt::calculate_tensor_pcc(a, b);

    // Torch's corrcoef doesn't accept two inputs, but expects two inputs to be stacked.
    auto stacked = at::stack({a.flatten(), b.flatten()});
    auto golden_pcc = at::min(at::corrcoef(stacked)).item<double>();

    std::cerr << "pcc: " << calculated_pcc << std::endl;
    std::cerr << "golden pcc: " << golden_pcc << std::endl;

    EXPECT_TRUE(torch::isclose(torch::tensor(calculated_pcc), torch::tensor(golden_pcc)).item<bool>());
}

TEST_P(VerifTest, test_has_special_values)
{
    auto dtype = GetParam();

    torch::Tensor a = torch::tensor({1.0, 2.0, 3.0}, at::TensorOptions().dtype(dtype));
    EXPECT_FALSE(tt::has_special_values(a));

    if (!at::isFloatingType(dtype))
    {
        // Cannot create NaN or Inf for integer types, so return.
        return;
    }

    a = torch::tensor({1.0, 2.0, std::numeric_limits<double>::infinity()}, at::TensorOptions().dtype(dtype));
    EXPECT_TRUE(tt::has_special_values(a));

    a = torch::tensor({1.0, 2.0, std::numeric_limits<double>::quiet_NaN()}, at::TensorOptions().dtype(dtype));
    EXPECT_TRUE(tt::has_special_values(a));
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
