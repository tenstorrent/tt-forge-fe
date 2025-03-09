// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
// Modified from https://github.com/CVMI-Lab/PAConv/tree/main/scene_seg/lib/paconv_lib/src/gpu

#include <torch/extension.h>
#include <torch/torch.h>

void assign_score_withk_forward_wrapper(
    int B,
    int N0,
    int N1,
    int M,
    int K,
    int O,
    int aggregate,
    const at::Tensor& points,
    const at::Tensor& centers,
    const at::Tensor& scores,
    const at::Tensor& knn_idx,
    at::Tensor& output);

void assign_score_withk_backward_wrapper(
    int B,
    int N0,
    int N1,
    int M,
    int K,
    int O,
    int aggregate,
    const at::Tensor& grad_out,
    const at::Tensor& points,
    const at::Tensor& centers,
    const at::Tensor& scores,
    const at::Tensor& knn_idx,
    at::Tensor& grad_points,
    at::Tensor& grad_centers,
    at::Tensor& grad_scores);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "assign_score_withk_forward_wrapper",
        &assign_score_withk_forward_wrapper,
        "Assign score kernel forward (GPU), save memory version");
    m.def(
        "assign_score_withk_backward_wrapper",
        &assign_score_withk_backward_wrapper,
        "Assign score kernel backward (GPU), save memory version");
}
