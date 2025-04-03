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
    // a = torch::tensor({1.0, 2.0, std::nan("")});
    // b = torch::tensor({1.0, 2.0, std::nan("")});
    // EXPECT_TRUE(tt::all_close(a, b, true /* equal_nan */));
}

INSTANTIATE_TEST_SUITE_P(
    VerifTest,
    VerifTest,
    ::testing::Values(at::kFloat, at::kDouble, at::kInt, at::kLong, at::kShort, at::kByte, at::kBool, at::kBFloat16));
// ::testing::PrintToStringParamName());

}  // namespace tt::test
