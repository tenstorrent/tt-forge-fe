# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import RegNetForImageClassification

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
from test.models.pytorch.vision.regnet.model_utils.image_utils import (
    preprocess_input_data,
)
from test.models.pytorch.vision.vision_utils.utils import load_vision_model_and_input

variants = [
    "facebook/regnet-y-040",
    "facebook/regnet-y-064",
    "facebook/regnet-y-080",
    "facebook/regnet-y-120",
    "facebook/regnet-y-160",
    "facebook/regnet-y-320",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_regnet_img_classification(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.REGNET,
        variant=variant,
        task=Task.CV_IMAGE_CLS,
        source=Source.HUGGINGFACE,
    )

    # Load the image processor and the RegNet model
    framework_model = RegNetForImageClassification.from_pretrained(variant, return_dict=False).to(torch.bfloat16)

    # Preprocess the image
    inputs = preprocess_input_data(variant)
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

    # Model Verification and inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # post processing
    logits = co_out[0]
    predicted_label = logits.argmax(-1).item()
    print(framework_model.config.id2label[predicted_label])


variants_with_weights = {
    "regnet_y_400mf": "RegNet_Y_400MF_Weights",
    "regnet_y_800mf": "RegNet_Y_800MF_Weights",
    "regnet_y_1_6gf": "RegNet_Y_1_6GF_Weights",
    "regnet_y_3_2gf": "RegNet_Y_3_2GF_Weights",
    "regnet_y_8gf": "RegNet_Y_8GF_Weights",
    "regnet_y_16gf": "RegNet_Y_16GF_Weights",
    "regnet_y_32gf": "RegNet_Y_32GF_Weights",
    "regnet_y_128gf": "RegNet_Y_128GF_Weights",
    "regnet_x_400mf": "RegNet_X_400MF_Weights",
    "regnet_x_800mf": "RegNet_X_800MF_Weights",
    "regnet_x_1_6gf": "RegNet_X_1_6GF_Weights",
    "regnet_x_3_2gf": "RegNet_X_3_2GF_Weights",
    "regnet_x_8gf": "RegNet_X_8GF_Weights",
    "regnet_x_16gf": "RegNet_X_16GF_Weights",
    "regnet_x_32gf": "RegNet_X_32GF_Weights",
}

variants = [
    "regnet_y_400mf",
    "regnet_y_800mf",
    "regnet_y_1_6gf",
    "regnet_y_3_2gf",
    "regnet_y_8gf",
    "regnet_y_16gf",
    "regnet_y_32gf",
    pytest.param("regnet_y_128gf", marks=pytest.mark.xfail(reason="Cannot fit in L1")),
    "regnet_x_400mf",
    "regnet_x_800mf",
    "regnet_x_1_6gf",
    "regnet_x_3_2gf",
    "regnet_x_8gf",
    "regnet_x_16gf",
    "regnet_x_32gf",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_regnet_torchvision(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.REGNET,
        variant=variant,
        task=Task.CV_IMAGE_CLS,
        source=Source.TORCHVISION,
    )

    # Load model and input
    weight_name = variants_with_weights[variant]
    framework_model, inputs = load_vision_model_and_input(variant, "classification", weight_name)
    framework_model.to(torch.bfloat16)
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
    if variant == "regnet_x_8gf":
        verify_cfg = VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95))

    # Model Verification and inference
    fw_out, co_out = verify(inputs, framework_model, compiled_model, verify_cfg=verify_cfg)

    # Run model on sample data and print results
    print_cls_results(fw_out[0], co_out[0])
