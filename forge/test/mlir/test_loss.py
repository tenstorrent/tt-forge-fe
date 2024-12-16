# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

import forge


@pytest.mark.parametrize(
    "prediction_shape",
    [
        (33,),
        (128,),
        (3, 5),
        (32, 32),
        (33, 127),
        (128, 20),
    ],
)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.push
def test_l1_loss(prediction_shape, reduction):
    forge_loss = forge.op.loss.L1Loss("l1_loss", reduction=reduction)
    torch_loss = torch.nn.L1Loss(reduction=reduction)

    prediction = torch.randn(prediction_shape, requires_grad=True)
    prediction_forge = forge.tensor.Tensor.create_from_torch(prediction)
    target = torch.randn((prediction_shape))
    target_forge = forge.tensor.Tensor.create_from_torch(target)

    forge_loss = forge.compile(forge_loss, sample_inputs=[prediction_forge, target_forge])
    forge_loss_out = forge_loss(prediction, target)
    torch_loss_out = torch_loss(prediction, target)

    assert torch.allclose(torch_loss_out, forge_loss_out[0], rtol=11e-3)


@pytest.mark.parametrize(
    "prediction_shape",
    [
        (3, 5),
        (32, 32),
        (33, 127),
        (128, 20),
    ],
)
@pytest.mark.push
def test_cross_entropy_loss(prediction_shape):
    forge_loss = forge.op.loss.CrossEntropyLoss("cross_entropy_loss")
    torch_loss = torch.nn.CrossEntropyLoss()

    prediction = torch.randn(prediction_shape, requires_grad=True)
    prediction_forge = forge.tensor.Tensor.create_from_torch(prediction)
    target = torch.empty(prediction_shape[0], dtype=torch.long).random_(prediction_shape[-1])
    target = nn.functional.one_hot(target, num_classes=prediction_shape[-1]).float()
    target_forge = forge.tensor.Tensor.create_from_torch(target)

    forge_loss = forge.compile(forge_loss, sample_inputs=[prediction_forge, target_forge])
    forge_loss_out = forge_loss(prediction, target)
    torch_loss_out = torch_loss(prediction, target)

    assert torch.allclose(torch_loss_out, forge_loss_out[0], rtol=11e-3)


@pytest.mark.parametrize(
    "prediction_shape",
    [
        (33,),
        (128,),
        (2, 2),
        (3, 5),
        (32, 32),
        (33, 127),
        (128, 20),
        (128, 128),
    ],
)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_mse_loss(prediction_shape, reduction):
    forge_loss = forge.op.loss.MSELoss("mse_loss", reduction=reduction)
    torch_loss = torch.nn.MSELoss(reduction=reduction)

    prediction = torch.randn(prediction_shape, requires_grad=True)
    prediction_forge = forge.tensor.Tensor.create_from_torch(prediction)
    target = torch.randn((prediction_shape))
    target_forge = forge.tensor.Tensor.create_from_torch(target)

    forge_loss = forge.compile(forge_loss, sample_inputs=[prediction_forge, target_forge])
    forge_loss_out = forge_loss(prediction, target)
    torch_loss_out = torch_loss(prediction, target)

    assert torch.allclose(
        torch_loss_out, forge_loss_out[0], rtol=5e-2, atol=5e-3
    )  # relative tolerance is 5% and absolute tolerance is 0.005


@pytest.mark.parametrize(
    "prediction_shape",
    [
        (33,),
        (128,),
        (2, 2),
        (3, 5),
        (32, 32),
        (33, 127),
        (128, 20),
        (128, 128),
    ],
)
def test_triplet_margin_loss(prediction_shape):
    forge_loss = forge.op.loss.TripletMarginLoss("triplet_margin_loss")
    torch_loss = torch.nn.TripletMarginLoss()

    anchor = torch.randn(prediction_shape, requires_grad=True)
    anchor_forge = forge.tensor.Tensor.create_from_torch(anchor)
    positive = torch.randn(prediction_shape, requires_grad=True)
    positive_forge = forge.tensor.Tensor.create_from_torch(positive)
    negative = torch.randn(prediction_shape, requires_grad=True)
    negative_forge = forge.tensor.Tensor.create_from_torch(negative)

    forge_loss = forge.compile(forge_loss, sample_inputs=[anchor_forge, positive_forge, negative_forge])
    forge_loss_out = forge_loss(anchor_forge, positive_forge, negative_forge)
    torch_loss_out = torch_loss(anchor, positive, negative)
    print(f"torch_loss_out: {torch_loss_out}")
    print(f"forge_loss_out: {forge_loss_out[0]} {forge_loss_out[0].shape}")

    assert torch.allclose(torch_loss_out, forge_loss_out[0], rtol=5e-2)
