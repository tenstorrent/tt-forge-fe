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
        (2, 2),
        (3, 5),
        (32, 32),
        (33, 127),
        (128, 20),
        (128, 128),
    ],
)
def test_nll_loss(prediction_shape):
    forge_loss = forge.op.loss.NLLLoss("nll_loss")
    torch_loss = torch.nn.NLLLoss()

    prediction = torch.randn(prediction_shape, requires_grad=True).to(torch.float32)
    prediction_forge = forge.tensor.Tensor.create_from_torch(prediction)
    target = torch.empty(prediction_shape[0], dtype=torch.long).random_(prediction_shape[-1])

    # Because of the following error
    # RuntimeError: TT_FATAL @ ../embedding_device_operation.cpp:28: weights.get_dtype() == DataType::BFLOAT16
    # We need to convert the target to one hot, which is different from torch
    target_one_hot = nn.functional.one_hot(target, num_classes=prediction_shape[-1]).float()

    target_forge = forge.tensor.Tensor.create_from_torch(target_one_hot)

    forge_loss = forge.compile(forge_loss, sample_inputs=[prediction_forge, target_forge])
    forge_loss_out = forge_loss(prediction, target_one_hot)
    torch_loss_out = torch_loss(prediction, target)

    assert torch.allclose(torch_loss_out, forge_loss_out[0], rtol=11e-3)
