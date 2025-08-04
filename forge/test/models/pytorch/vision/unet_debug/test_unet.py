# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

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
from forge.verify.verify import verify
from third_party.tt_forge_models.tools.utils import get_file

from test.utils import download_model


def generate_model_unet_imgseg_torchhub_pytorch(variant):
    model = download_model(
        torch.hub.load,
        "mateuszbuda/brain-segmentation-pytorch",
        variant,
        in_channels=3,
        out_channels=1,
        init_features=32,
        pretrained=True,
    )
    model.eval()

    # Download an example input image
    file_path = get_file(
        "https://github.com/mateuszbuda/brain-segmentation-pytorch/raw/master/assets/TCGA_CS_4944.png",
    )
    input_image = Image.open(file_path)
    m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=m, std=s),
        ]
    )
    input_tensor = preprocess(input_image)
    img_batch = input_tensor.unsqueeze(0)

    return model.to(torch.bfloat16), [img_batch.to(torch.bfloat16)], {}


@pytest.mark.nightly
def test_unet_torchhub_pytorch():

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH, model=ModelArch.UNET, source=Source.TORCH_HUB, task=Task.IMAGE_SEGMENTATION
    )

    framework_model, inputs, _ = generate_model_unet_imgseg_torchhub_pytorch(
        "unet",
    )

    # logger.info("framework_model={}",framework_model)
    # logger.info("framework_model.__class__={}",framework_model.__class__)

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # with torch.no_grad():
    #     op = framework_model(*inputs)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model)
