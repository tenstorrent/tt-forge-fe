# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import random

import pytest
import timm
import torch
from datasets import load_dataset
from tabulate import tabulate
from transformers import AutoImageProcessor, ResNetForImageClassification

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

from test.models.pytorch.vision.vision_utils.utils import load_vision_model_and_input
from test.utils import download_model

variants = [
    "microsoft/resnet-50",
]


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_resnet_hf(variant):
    random.seed(0)

    # Record model details
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.RESNET,
        variant="50",
        source=Source.HUGGINGFACE,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load tiny dataset
    dataset = load_dataset("zh-plus/tiny-imagenet")
    images = random.sample(dataset["valid"]["image"], 10)

    # Load framework model
    framework_model = download_model(ResNetForImageClassification.from_pretrained, variant, return_dict=False).to(
        torch.bfloat16
    )

    # Compile model
    input_sample = [torch.rand(1, 3, 224, 224).to(torch.bfloat16)]

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
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95), verify_emitc_correctness=True),
    )

    # Run model on sample data and print results
    run_and_print_results(framework_model, compiled_model, images)


def run_and_print_results(framework_model, compiled_model, inputs):
    """
    Runs inference using both a framework model and a compiled model on a list of input images,
    then prints the results in a formatted table.

    Args:
        framework_model: The original framework-based model.
        compiled_model: The compiled version of the model.
        inputs: A list of images to process and classify.
    """
    label_dict = framework_model.config.id2label
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

    results = []
    for i, image in enumerate(inputs):
        processed_inputs = processor(image, return_tensors="pt")["pixel_values"].to(torch.bfloat16)

        cpu_logits = framework_model(processed_inputs)[0]
        cpu_conf, cpu_idx = cpu_logits.softmax(-1).max(-1)
        cpu_pred = label_dict.get(cpu_idx.item(), "Unknown")

        tt_logits = compiled_model(processed_inputs)[0]
        tt_conf, tt_idx = tt_logits.softmax(-1).max(-1)
        tt_pred = label_dict.get(tt_idx.item(), "Unknown")

        results.append([i + 1, cpu_pred, cpu_conf.item(), tt_pred, tt_conf.item()])

    print(
        tabulate(
            results,
            headers=["Example", "CPU Prediction", "CPU Confidence", "Compiled Prediction", "Compiled Confidence"],
            tablefmt="grid",
        )
    )


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

    # Verify data on sample input
    verify(
        input_sample,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)),
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

    verify_cfg = VerifyConfig()
    if variant == "resnet34":
        verify_cfg = VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.98))

    # Model Verification
    verify(inputs, framework_model, compiled_model, verify_cfg=verify_cfg)
