# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import random

import pytest
import timm
import torch
from datasets import load_dataset
from tabulate import tabulate
from torchvision.models.resnet import resnet50
from transformers import AutoImageProcessor, ResNetForImageClassification

import forge
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

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
    record_forge_property("model_name", module_name)

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
    record_forge_property("model_name", module_name)

    # Load framework model
    framework_model = download_model(timm.create_model, "resnet50", pretrained=True)

    # Compile model
    input_sample = [torch.rand(1, 3, 224, 224)]
    compiled_model = forge.compile(framework_model, sample_inputs=input_sample, module_name=module_name)

    # Verify data on sample input
    verify(input_sample, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)))


@pytest.mark.nightly
def test_resnet_torchvision(record_forge_property):
    # Record model details
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="resnet",
        source=Source.TORCHVISION,
        variant="50",
        task=Task.IMAGE_CLASSIFICATION,
    )
    record_forge_property("model_name", module_name)

    # Load framework model
    framework_model = resnet50()

    # Compile model
    input_sample = [torch.rand(1, 3, 224, 224)]
    compiled_model = forge.compile(framework_model, input_sample)

    # Verify data on sample input
    verify(input_sample, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)))
