// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>

#include "test/common.hpp"
#include "verif/verif_ops.hpp"

namespace tt::test
{

class VerifOpsTest : public ::testing::Test, public ::testing::WithParamInterface<at::ScalarType>
{
   public:
    virtual void SetUp() override
    {
        // Reset the torch seed for each test.
        torch::manual_seed(46);
    }
    virtual void TearDown() override {}

   protected:
    torch::Tensor create_random_tensor(const std::vector<int64_t>& shape)
    {
        auto dtype = GetParam();
        if (torch::isFloatingType(dtype))
        {
            return torch::randn(shape, at::TensorOptions().dtype(dtype));
        }
        else
        {
            return torch::randint(0, 1000, shape, at::TensorOptions().dtype(dtype));
        }
    }

    torch::Tensor create_nan_or_zero_tensor(const std::vector<int64_t>& shape)
    {
        auto dtype = GetParam();
        if (torch::isFloatingType(dtype))
        {
            return torch::full(shape, std::numeric_limits<double>::quiet_NaN(), at::TensorOptions().dtype(dtype));
        }
        else
        {
            return torch::full(shape, 0, at::TensorOptions().dtype(dtype));
        }
    }
};

class VerifPCCTest : public VerifOpsTest
{
};

class VerifAllCloseTest : public VerifOpsTest
{
};

class VerifMaxAbsDiffTest : public VerifOpsTest
{
};

class VerifHasSpecialValuesTest : public VerifOpsTest
{
};

TEST_P(VerifAllCloseTest, test_sanity)
{
    auto dtype = GetParam();
    torch::Tensor a = torch::tensor({1.0, 2.0, 3.0}, at::TensorOptions().dtype(dtype));
    torch::Tensor b = torch::tensor({1.0, 2.0, 3.0}, at::TensorOptions().dtype(dtype));
    EXPECT_TRUE(tt::all_close(a, b));

    a = torch::tensor({1.0, 2.0, 3.0}, at::TensorOptions().dtype(dtype));
    b = torch::tensor({0.0, 2.0, 3.0}, at::TensorOptions().dtype(dtype));
    EXPECT_FALSE(tt::all_close(a, b));
}

TEST_P(VerifAllCloseTest, test_large_tensors)
{
    torch::Tensor a = create_random_tensor({1000, 1000});
    torch::Tensor b = create_random_tensor({1000, 1000});

    EXPECT_EQ(torch::allclose(a, b), tt::all_close(a, b));
}

// Test correctness when flattened input tensors are non-contiguous.
TEST_P(VerifAllCloseTest, test_non_contiguous)
{
    // To test correctness in case of non-contiguous tensors, we will
    // create two tensors of size 64000x20 and select the 13th column from both tensors; these inputs when flattened
    // will have stride != 1, i.e. they will be non-contiguous.
    constexpr int64_t column = 13;
    torch::Tensor a = create_random_tensor({64000, 20});
    torch::Tensor b = create_random_tensor({64000, 20});

    // Set each element of the selected columns so that we all close to be true.
    for (int64_t i = 0; i < a.size(0); ++i)
    {
        a[i][column] = 46;
        b[i][column] = 46;
    }

    // Select the wanted column from both tensors - the resulting tensors will have stride != 1.
    auto a_view = a.select(1, column);
    auto b_view = b.select(1, column);

    EXPECT_TRUE(tt::all_close(a_view, b_view));
}

TEST_P(VerifPCCTest, test_degenerate_cases)
{
    auto dtype = GetParam();

    torch::Tensor a = torch::tensor({1.0, 2.0, 3.0}, at::TensorOptions().dtype(dtype));
    torch::Tensor b = torch::tensor({1.0, 2.0, 3.0}, at::TensorOptions().dtype(dtype));
    double pcc = tt::calculate_tensor_pcc(a, b);
    EXPECT_TRUE(torch::allclose(torch::tensor(pcc), torch::tensor(1.0)));

    a = torch::tensor({1.0, 2.0, 3.0}, at::TensorOptions().dtype(dtype));
    b = torch::tensor({3.0, 2.0, 1.0}, at::TensorOptions().dtype(dtype));
    pcc = tt::calculate_tensor_pcc(a, b);
    EXPECT_TRUE(torch::allclose(torch::tensor(pcc), torch::tensor(-1.0)));

    a = torch::tensor({1.0, 1.0, 1.0}, at::TensorOptions().dtype(dtype));
    b = torch::tensor({1.0, 1.0, 1.0}, at::TensorOptions().dtype(dtype));
    pcc = tt::calculate_tensor_pcc(a, b);
    EXPECT_TRUE(std::isnan(pcc));

    a = torch::tensor({1.0, 1.0, 1.0}, at::TensorOptions().dtype(dtype));
    b = torch::tensor({2.0, 2.0, 2.0}, at::TensorOptions().dtype(dtype));
    pcc = tt::calculate_tensor_pcc(a, b);
    EXPECT_TRUE(std::isnan(pcc));
}

TEST_P(VerifPCCTest, test_large_tensors)
{
    torch::Tensor a = create_random_tensor({1000, 1000});
    torch::Tensor b = create_random_tensor({1000, 1000});

    auto calculated_pcc = tt::calculate_tensor_pcc(a, b);

    // Torch's corrcoef doesn't accept two inputs, but expects two inputs to be stacked.
    auto stacked = at::stack({a.flatten().to(c10::kDouble), b.flatten().to(c10::kDouble)});
    auto golden_pcc = at::min(at::corrcoef(stacked)).item<double>();

    EXPECT_TRUE(torch::allclose(torch::tensor(calculated_pcc), torch::tensor(golden_pcc)));
}

// Test correctness when flattened input tensors are non-contiguous.
TEST_P(VerifPCCTest, test_non_contiguous)
{
    // To test correctness in case of non-contiguous tensors, we will
    // create two tensors of size 64000x20 and select the 13th column from both tensors; these inputs when flattened
    // will have stride != 1, i.e. they will be non-contiguous.
    constexpr int64_t column = 13;
    torch::Tensor a = create_random_tensor({64000, 20});
    torch::Tensor b = create_random_tensor({64000, 20});

    // Set each element of the selected columns so that we expect pcc == 1.0.
    for (int64_t i = 0; i < a.size(0); ++i)
    {
        a[i][column] = i % 46;
        b[i][column] = i % 46;
    }

    // Select the wanted column from both tensors - the resulting tensors will have stride != 1.
    auto a_view = a.select(1, column);
    auto b_view = b.select(1, column);

    double pcc = tt::calculate_tensor_pcc(a_view, b_view);

    EXPECT_TRUE(torch::allclose(torch::tensor(pcc), torch::tensor(1.0)));
}

TEST_P(VerifHasSpecialValuesTest, test_sanity)
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

TEST_P(VerifHasSpecialValuesTest, test_large_tensor)
{
    torch::Tensor a = torch::rand({1000, 1000}, at::TensorOptions().dtype(at::kFloat));

    // Generate 10 random indices and set tensor at those indices to NaN.
    for (int i = 0; i < 10; ++i)
    {
        int x = torch::randint(0, 1000, {1}).item<int>();
        int y = torch::randint(0, 1000, {1}).item<int>();

        a[x][y] = std::numeric_limits<double>::quiet_NaN();
    }

    EXPECT_TRUE(tt::has_special_values(a));
}

// Test correctness when flattened input tensors are non-contiguous.
TEST_P(VerifHasSpecialValuesTest, test_non_contiguous)
{
    // To test correctness in case of non-contiguous tensors, we will
    // create two tensors of size 64000x20 and select the 13th column from both tensors; these inputs when flattened
    // will have stride != 1, i.e. they will be non-contiguous.
    constexpr int64_t column = 13;
    torch::Tensor a = create_nan_or_zero_tensor({64000, 20});

    // Set each element of the selected columns so that they are not NaN.
    for (int64_t i = 0; i < a.size(0); ++i)
    {
        a[i][column] = 46;
    }

    // Select the wanted column from both tensors - the resulting tensors will have stride != 1.
    auto a_view = a.select(1, column);

    EXPECT_FALSE(tt::has_special_values(a_view));
}

TEST_P(VerifMaxAbsDiffTest, test_sanity)
{
    auto dtype = GetParam();

    torch::Tensor a = torch::tensor({1.0, 2.0, 3.0}, at::TensorOptions().dtype(dtype));
    torch::Tensor b = torch::tensor({1.0, 2.0, 3.0}, at::TensorOptions().dtype(dtype));
    EXPECT_TRUE(tt::max_abs_diff(a, b) == 0);

    a = torch::tensor({1.0, 2.0, 3.0}, at::TensorOptions().dtype(dtype));
    b = torch::tensor({0.0, 2.0, 3.0}, at::TensorOptions().dtype(dtype));
    EXPECT_TRUE(tt::max_abs_diff(a, b) == 1);
}

// Test correctness when flattened input tensors are non-contiguous.
TEST_P(VerifMaxAbsDiffTest, test_non_contiguous)
{
    // To test correctness in case of non-contiguous tensors, we will
    // create two tensors of size 64000x20 and select a single column from both tensors; these inputs when flattened
    // will have stride != 1, i.e. they will be non-contiguous.
    torch::Tensor a = torch::rand({64000, 20}, at::TensorOptions().dtype(at::kFloat));
    torch::Tensor b = torch::rand({64000, 20}, at::TensorOptions().dtype(at::kFloat));

    // Set each element of the selected columns so that we expect max_abs_diff = 46.0.
    constexpr int64_t column = 13;
    for (int64_t i = 0; i < a.size(0); ++i)
    {
        a[i][column] = 13;
        b[i][column] = 59;
    }

    // Select the wanted column from both tensors - the resulting tensors will have stride != 1.
    auto a_view = a.select(1, column);
    auto b_view = b.select(1, column);

    double max_abs_diff = tt::max_abs_diff(a_view, b_view);

    EXPECT_TRUE(torch::allclose(torch::tensor(max_abs_diff), torch::tensor(46.0)));
}

TEST_P(VerifMaxAbsDiffTest, test_large_tensors)
{
    torch::Tensor a = torch::rand({1000, 1000}, at::TensorOptions().dtype(at::kFloat));
    torch::Tensor b = torch::rand({1000, 1000}, at::TensorOptions().dtype(at::kFloat));

    EXPECT_EQ(torch::max(torch::abs(a - b)).item<double>(), tt::max_abs_diff(a, b));
}

INSTANTIATE_TEST_SUITE_P(
    VerifPCCTest,
    VerifPCCTest,
    ::testing::Values(at::kFloat, at::kBFloat16, at::kHalf, at::kDouble, at::kInt, at::kLong, at::kShort));

INSTANTIATE_TEST_SUITE_P(
    VerifAllCloseTest,
    VerifAllCloseTest,
    ::testing::Values(at::kFloat, at::kBFloat16, at::kHalf, at::kDouble, at::kInt, at::kLong, at::kShort));

INSTANTIATE_TEST_SUITE_P(
    VerifMaxAbsDiffTest,
    VerifMaxAbsDiffTest,
    ::testing::Values(at::kFloat, at::kBFloat16, at::kHalf, at::kDouble, at::kInt, at::kLong, at::kShort));

INSTANTIATE_TEST_SUITE_P(
    VerifHasSpecialValuesTest,
    VerifHasSpecialValuesTest,
    ::testing::Values(at::kFloat, at::kBFloat16, at::kHalf, at::kDouble, at::kInt, at::kLong, at::kShort));

}  // namespace tt::test
