# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import timm
import torch
from PIL import Image
from third_party.tt_forge_models.tools.utils import get_file
from third_party.tt_forge_models.xception.pytorch import ModelLoader, ModelVariant
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

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

from test.utils import download_model


def generate_model_xception_imgcls_timm(variant):
    # STEP 2: Create Forge module from PyTorch model
    framework_model = download_model(timm.create_model, variant, pretrained=True)
    framework_model.eval()

    # STEP 3: Prepare input
    config = resolve_data_config({}, model=framework_model)
    transform = create_transform(**config)
    file_path = get_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg")
    img = Image.open(file_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    return framework_model, [img_tensor]


variants = [
    ModelVariant.XCEPTION41,
    ModelVariant.XCEPTION65,
    ModelVariant.XCEPTION71,
    ModelVariant.XCEPTION71_TF_IN1K,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_xception_timm(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.XCEPTION,
        variant=variant,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load model and inputs
    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model(dtype_override=torch.bfloat16)
    input_tensor = loader.load_inputs(dtype_override=torch.bfloat16)
    inputs = [input_tensor]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing
    loader.print_cls_results(co_out)
