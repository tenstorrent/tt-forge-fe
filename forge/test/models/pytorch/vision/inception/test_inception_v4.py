# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
## Inception V4
import os
import pytest

import torch

from pytorchcv.model_provider import get_model as ptcv_get_model

import forge

from test.utils import download_model
from test.models.pytorch.vision.inception.utils.model_utils import get_image, preprocess_timm_model
from test.models.utils import build_module_name

torch.multiprocessing.set_sharing_strategy("file_system")


def generate_model_inceptionV4_imgcls_osmr_pytorch(variant):
    # Load model
    framework_model = download_model(ptcv_get_model, variant, pretrained=True)

    # Load and pre-process image
    img_tensor = get_image()

    return framework_model, [img_tensor]


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_inception_v4_osmr_pytorch(record_forge_property):
    model, inputs = generate_model_inceptionV4_imgcls_osmr_pytorch("inceptionv4")
    module_name = build_module_name(framework="pt", model="inception_v4_pytorch", task="osmr")
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)


def generate_model_inceptionV4_imgcls_timm_pytorch(variant):
    # Load model & Preprocess image
    framework_model, img_tensor = download_model(preprocess_timm_model, variant)
    return framework_model, [img_tensor]


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_inception_v4_timm_pytorch(record_forge_property):
    model, inputs = generate_model_inceptionV4_imgcls_timm_pytorch("inception_v4")

    module_name = build_module_name(framework="pt", model="inception_v4", task="osmr", source="timm")
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)
