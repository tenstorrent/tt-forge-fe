# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import shutil
import zipfile

import pytest
import requests
import torch

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import Framework, Source, Task, record_model_properties
from forge.verify.verify import verify

from test.models.pytorch.vision.retinanet.model_utils.image_utils import img_preprocess
from test.models.pytorch.vision.retinanet.model_utils.model import Model
from test.models.pytorch.vision.vision_utils.utils import load_vision_model_and_input

variants = [
    "retinanet_rn18fpn",
    "retinanet_rn34fpn",
    "retinanet_rn50fpn",
    "retinanet_rn101fpn",
    "retinanet_rn152fpn",
]


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants)
def test_retinanet(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model="retinanet",
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.OBJECT_DETECTION,
    )

    # Prepare model
    url = f"https://github.com/NVIDIA/retinanet-examples/releases/download/19.04/{variant}.zip"
    local_zip_path = f"{variant}.zip"

    response = requests.get(url)
    with open(local_zip_path, "wb") as f:
        f.write(response.content)

    # Unzip the file
    with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
        zip_ref.extractall(".")

    # Find the path of the .pth file
    extracted_path = f"{variant}"
    checkpoint_path = ""
    for root, _, files in os.walk(extracted_path):
        for file in files:
            if file.endswith(".pth"):
                checkpoint_path = os.path.join(root, file)

    framework_model = Model.load(checkpoint_path)
    framework_model.eval()
    framework_model.to(torch.bfloat16)

    # Prepare input
    input_batch = img_preprocess()
    inputs = [input_batch.to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model)

    # Delete the extracted folder and the zip file
    shutil.rmtree(extracted_path)
    os.remove(local_zip_path)


variants_with_weights = {
    "retinanet_resnet50_fpn_v2": "RetinaNet_ResNet50_FPN_V2_Weights",
}


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants_with_weights.keys())
def test_retinanet_torchvision(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model="retinanet",
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCHVISION,
    )

    # Load model and input
    weight_name = variants_with_weights[variant]
    framework_model, inputs = load_vision_model_and_input(variant, "detection", weight_name)

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model)
