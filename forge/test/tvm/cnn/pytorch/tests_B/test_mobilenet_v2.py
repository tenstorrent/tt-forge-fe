# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import torch

from forge.config import CompileDepth, _get_global_compiler_config

from transformers import MobileNetV2FeatureExtractor, MobileNetV2ForSemanticSegmentation
from transformers import AutoImageProcessor

from forge import (
    PyTorchModule,
    VerifyConfig,
)
from forge.verify.backend import verify_module
from forge.verify.config import TestKind
from forge import DataFormat
from test.utils import download_model


def test_mobilenetv2_pytorch(test_kind, test_device):
    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    if test_kind.is_training():
        pytest.skip()  # Backward is currently unsupported

    # tenstorrent/forge#392
    import os
    os.environ["FORGE_DISABLE_CONSTANT_FOLDING"] = "1"
    
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    model = download_model(torch.hub.load, 
        "pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=True
    )
    module = PyTorchModule("mobilenetv2", model)

    if test_device.is_silicon():
        pcc = 0.95
    else:
        pcc = 0.99

    input_shape = (1, 3, 224, 224)
    
    # NOTE: On silicon, this model has a higher PCC when compared to framework when using Float16 (rather than Float16_b) as the fp32_fallback
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            pcc=pcc,
            fp32_fallback=DataFormat.Float16
        ),
    )

def test_mobilenetv2_deeplab(test_kind, test_device):
    
    pytest.skip("Needs padding")
    
    if (
        test_kind == TestKind.TRAINING
    ):  # Always run with recompute in post-commit CI. Nightly tests both
        pytest.skip()

    if test_kind.is_training():
        pytest.skip()  # Backward is currently unsupported

    import os
    os.environ["FORGE_PAD_SPARSE_MM"] = "{25:26}"
    
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.balancer_policy = "CNN"

    model = download_model(MobileNetV2ForSemanticSegmentation.from_pretrained, "Matthijs/deeplabv3_mobilenet_v2_1.0_513")
    module = PyTorchModule("mobilenetv2_deeplab", model)

    input_shape = (1, 3, 224, 224)
    
    # NOTE: On silicon, this model has a higher PCC when compared to framework when using Float16 (rather than Float16_b) as the fp32_fallback
    verify_module(
        module,
        (input_shape,),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            test_kind=test_kind,
            fp32_fallback=DataFormat.Float16
        ),
    )