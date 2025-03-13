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
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

from test.models.pytorch.vision.utils.utils import load_vision_model_and_input
from test.models.utils import Framework, Source, Task, build_module_name
from test.utils import download_model

variants = [
    "microsoft/resnet-50",
]


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_resnet_hf(variant, record_forge_property):
    random.seed(0)

    # Record model details
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="resnet",
        variant="50",
        source=Source.HUGGINGFACE,
        task=Task.IMAGE_CLASSIFICATION,
    )
    record_forge_property("tags.model_name", module_name)

    # Load tiny dataset
    dataset = load_dataset("zh-plus/tiny-imagenet")
    images = random.sample(dataset["valid"]["image"], 10)

    # Load framework model
    framework_model = download_model(ResNetForImageClassification.from_pretrained, variant, return_dict=False)

    # Compile model
    input_sample = [torch.rand(1, 3, 224, 224)]
    compiled_model = forge.compile(framework_model, input_sample)

    # Verify data on sample input
    verify(input_sample, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)))

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
        processed_inputs = processor(image, return_tensors="pt")["pixel_values"]

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
def test_resnet_timm(record_forge_property):
    # Record model details
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="resnet", source=Source.TIMM, variant="50", task=Task.IMAGE_CLASSIFICATION
    )
    record_forge_property("tags.model_name", module_name)

    # Load framework model
    framework_model = download_model(timm.create_model, "resnet50", pretrained=True)

    # Compile model
    input_sample = [torch.rand(1, 3, 224, 224)]
    compiled_model = forge.compile(framework_model, sample_inputs=input_sample, module_name=module_name)

    # Verify data on sample input
    verify(input_sample, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)))


variants_with_weights = {
    "resnet18": "ResNet18_Weights",
    "resnet34": "ResNet34_Weights",
    "resnet50": "ResNet50_Weights",
    "resnet101": "ResNet101_Weights",
    "resnet152": "ResNet152_Weights",
}


@pytest.mark.nightly
@pytest.mark.xfail(
    reason="RuntimeError: Tensor 0 - stride mismatch: expected [150528, 50176, 224, 1], got [3, 1, 672, 3]"
)
@pytest.mark.parametrize("variant", variants_with_weights.keys())
def test_resnet_torchvision(record_forge_property, variant):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="resnet",
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCHVISION,
    )

    # Record Forge Property
    record_forge_property("group", "generality")
    record_forge_property("tags.model_name", module_name)

    # Load model and input
    weight_name = variants_with_weights[variant]
    framework_model, inputs = load_vision_model_and_input(variant, "classification", weight_name)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
