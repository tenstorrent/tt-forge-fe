# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import requests
import torch
import torchvision.transforms as transforms
from PIL import Image

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

from test.models.pytorch.vision.monodle.model_utils.model import CenterNet3D


@pytest.mark.nightly
@pytest.mark.skip(reason="Floating point exception(core dumped)")
def test_monodle_pytorch():
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH, model=ModelArch.MONODLE, source=Source.TORCHVISION, task=Task.OBJECT_DETECTION
    )

    # Load data sample
    url = "https://images.rawpixel.com/image_1300/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L3BkMTA2LTA0Ny1jaGltXzEuanBn.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # Preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_tensor = transform(image).unsqueeze(0)

    framework_model = CenterNet3D(backbone="dla34").to(torch.bfloat16)
    framework_model.eval()

    inputs = [img_tensor.to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, compiler_cfg=compiler_cfg
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model)
