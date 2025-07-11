# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

from test.models.models_utils import print_cls_results
from test.models.pytorch.vision.mobilenet.model_utils.mobilenet_v3_ssd_utils import (
    load_input,
    load_model,
)

variants_with_weights = {
    "resnet18": "ResNet18_Weights",
    "resnet34": "ResNet34_Weights",
    "resnet50": "ResNet50_Weights",
    "resnet101": "ResNet101_Weights",
    "resnet152": "ResNet152_Weights",
}


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants_with_weights.keys())
def test_mobilenetv3_ssd(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.MOBILENETV3SSD,
        variant=variant,
        source=Source.TORCHVISION,
        task=Task.CV_IMAGE_CLS,
    )

    # Load model and input
    weight_name = variants_with_weights[variant]
    framework_model = load_model(variant, weight_name).to(torch.bfloat16)
    inputs = load_input()
    inputs = [inputs[0].to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    verify_cfg = VerifyConfig()
    if variant in ["resnet50", "resnet101"]:
        verify_cfg = VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95))
    elif variant == "resnet34":
        verify_cfg = VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.98))

    # Model Verification and Inference
    fw_out, co_out = verify(inputs, framework_model, compiled_model, verify_cfg=verify_cfg)

    # Post processing
    print_cls_results(fw_out[0], co_out[0])
