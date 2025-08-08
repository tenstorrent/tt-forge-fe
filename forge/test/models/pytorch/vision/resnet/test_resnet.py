# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import random

import pytest
import timm
import torch
from datasets import load_dataset
from third_party.tt_forge_models.resnet.pytorch import ModelLoader

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    ModelPriority,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

from test.models.models_utils import print_cls_results
from test.models.pytorch.vision.vision_utils.utils import load_vision_model_and_input
from test.utils import download_model


@pytest.mark.push
@pytest.mark.nightly
def test_resnet_hf():
    random.seed(0)

    # Record model details
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.RESNET,
        variant="50",
        source=Source.HUGGINGFACE,
        task=Task.IMAGE_CLASSIFICATION,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    # Load tiny dataset
    dataset = load_dataset("zh-plus/tiny-imagenet")
    images = random.sample(dataset["valid"]["image"], 10)

    # Load model and inputs
    loader = ModelLoader()
    framework_model = loader.load_model(dtype_override=torch.bfloat16)
    input_tensor = loader.load_inputs(dtype_override=torch.bfloat16)
    input_sample = [input_tensor]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    compiled_model = forge.compile(
        framework_model,
        input_sample,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Verify data on sample input
    verify(
        input_sample,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)),
    )

    # Run model on sample data and print results
    loader.run_and_print_results(framework_model, compiled_model, images)


@pytest.mark.nightly
def test_resnet_timm():
    # Record model details
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.RESNET,
        source=Source.TIMM,
        variant="50",
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load framework model
    framework_model = download_model(timm.create_model, "resnet50", pretrained=True).to(torch.bfloat16)

    # Compile model
    input_sample = [torch.rand(1, 3, 224, 224).to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    compiled_model = forge.compile(
        framework_model,
        sample_inputs=input_sample,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification and Inference
    fw_out, co_out = verify(
        input_sample,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)),
    )

    # Run model on sample data and print results
    print_cls_results(fw_out[0], co_out[0])


variants_with_weights = {
    "resnet18": "ResNet18_Weights",
    "resnet34": "ResNet34_Weights",
    "resnet50": "ResNet50_Weights",
    "resnet101": "ResNet101_Weights",
    "resnet152": "ResNet152_Weights",
}


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants_with_weights.keys())
def test_resnet_torchvision(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.RESNET,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
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

    pcc = 0.99
    if variant == "resnet34":
        pcc = 0.98
    elif variant in ["resnet50", "resnet152"]:
        pcc = 0.95

    # Model Verification and Inference
    fw_out, co_out = verify(
        inputs, framework_model, compiled_model, verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc))
    )

    # Run model on sample data and print results
    print_cls_results(fw_out[0], co_out[0])
