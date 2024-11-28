# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.model_demos.models.ghostnet import generate_model_ghostnet_imgcls_timm
import forge
import torch
from forge.op.eval.common import compare_with_golden_pcc

variants = ["ghostnet_100"]


@pytest.mark.xfail(reason="Runtime error : Invalid arguments to reshape")
@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.nightly
def test_ghostnet_timm(variant, test_device):
    model, inputs = generate_model_ghostnet_imgcls_timm(variant)
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=f"pt_{variant}")

    co_out = compiled_model(*inputs)
    fw_out = model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out

    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])
