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
def test_kl_div_loss(prediction_shape, reduction):
    forge_loss = forge.op.loss.KLDivLoss("kl_div_loss", reduction=reduction)
    torch_loss = torch.nn.KLDivLoss(reduction=reduction)

    prediction = nn.functional.log_softmax(torch.randn(prediction_shape, requires_grad=True), dim=-1)
    prediction_forge = forge.tensor.Tensor.create_from_torch(prediction)
    target = torch.randn(prediction_shape)
    # softmax the target
    target = nn.functional.softmax(target, dim=-1)
    target_forge = forge.tensor.Tensor.create_from_torch(target)

    forge_loss = forge.compile(forge_loss, sample_inputs=[prediction_forge, target_forge])
    forge_loss_out = forge_loss(prediction, target)[0]
    torch_loss_out = torch_loss(prediction, target)
    assert torch.allclose(torch_loss_out, forge_loss_out, rtol=5e-2)


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
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_nll_loss(prediction_shape, reduction):
    forge_loss = forge.op.loss.NLLLoss("nll_loss", reduction=reduction)
    torch_loss = torch.nn.NLLLoss(reduction=reduction)

    prediction = torch.randn(prediction_shape, requires_grad=True)
    prediction = nn.functional.log_softmax(prediction, dim=-1)
    prediction_forge = forge.tensor.Tensor.create_from_torch(prediction)

    batch_size = prediction_shape[0] if len(prediction_shape) > 1 else 1
    target = torch.randint(0, prediction_shape[-1], (batch_size,), dtype=torch.long)

    # Because of the following error
    # RuntimeError: TT_FATAL @ ../embedding_device_operation.cpp:28: weights.get_dtype() == DataType::BFLOAT16
    # We need to convert the target to one hot, which is different from torch
    # https://github.com/tenstorrent/tt-mlir/issues/1503
    target_one_hot = nn.functional.one_hot(target, num_classes=prediction_shape[-1]).float()

    if batch_size == 1:  # Handle 1D case, remove the batch dimension
        target_one_hot = target_one_hot.squeeze(0)
        target = target.squeeze(0)

    target_forge = forge.tensor.Tensor.create_from_torch(target_one_hot)

    forge_loss = forge.compile(forge_loss, sample_inputs=[prediction_forge, target_forge])
    forge_loss_out = forge_loss(prediction, target_one_hot)
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
def test_huber_loss(prediction_shape, reduction):
    forge_loss = forge.op.loss.HuberLoss("huber_loss", delta=1.0, reduction=reduction)
    torch_loss = torch.nn.HuberLoss(reduction=reduction, delta=1.0)

    prediction = torch.randn(prediction_shape, requires_grad=True)
    prediction_forge = forge.tensor.Tensor.create_from_torch(prediction)
    target = torch.randn(prediction_shape)
    target_forge = forge.tensor.Tensor.create_from_torch(target)

    forge_loss = forge.compile(forge_loss, sample_inputs=[prediction_forge, target_forge])
    forge_loss_out = forge_loss(prediction, target)
    torch_loss_out = torch_loss(prediction, target)

    assert torch.allclose(torch_loss_out, forge_loss_out[0], rtol=5e-2)
