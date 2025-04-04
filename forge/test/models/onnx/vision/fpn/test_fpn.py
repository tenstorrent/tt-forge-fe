# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import onnx
import os
import pytest

# TODO: These are old forge, we should update them to the currently version.
# import forge
# from forge.verify.backend import verify_module
# from forge import DepricatedVerifyConfig


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="Requires restructuring")
@pytest.mark.nightly
def test_fpn_onnx(test_device, test_kind):
    compiler_cfg = forge.config.CompilerConfig()
    compiler_cfg.default_df_override = forge._C.Float16_b

    # Load FPN model
    onnx_model_path = "third_party/confidential_customer_models/generated/files/fpn.onnx"
    model = onnx.load(onnx_model_path)
    tt_model = forge.OnnxModule("onnx_fpn", model)

    feat0 = torch.rand(1, 10, 64, 64)
    feat1 = torch.rand(1, 20, 16, 16)
    feat2 = torch.rand(1, 30, 8, 8)

    verify_module(
        tt_model,
        input_shapes=[feat0.shape, feat1.shape, feat2.shape],
        inputs=[(feat0, feat1, feat2)],
        verify_cfg=DepricatedVerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=test_kind,
        ),
    )
