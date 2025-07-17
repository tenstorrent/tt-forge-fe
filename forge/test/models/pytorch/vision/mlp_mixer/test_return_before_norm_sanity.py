# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import timm
import torch
from loguru import logger
from PIL import Image
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
from third_party.tt_forge_models.tools.utils import get_file

from test.models.models_utils import print_cls_results
from test.utils import download_model


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.stem = model.stem
        self.b0 = model.blocks

    def forward(self, x):
        x = self.stem(x)
        y = self.b0(x)
        return y


varaints = [
    pytest.param(
        "mixer_b16_224",
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", varaints)
def test_mlp_mixer_timm_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.MLPMIXER,
        variant=variant,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
    )

    load_pretrained_weights = True
    if variant in ["mixer_s32_224", "mixer_s16_224", "mixer_b32_224", "mixer_l32_224"]:
        load_pretrained_weights = False

    framework_model = download_model(timm.create_model, variant, pretrained=load_pretrained_weights).to(torch.bfloat16)
    framework_model = Wrapper(framework_model).to(torch.bfloat16)

    config = resolve_data_config({}, model=framework_model)
    transform = create_transform(**config)

    try:
        if variant in [
            "mixer_b16_224_in21k",
            "mixer_b16_224_miil_in21k",
            "mixer_l16_224_in21k",
            "mixer_b16_224.goog_in21k",
        ]:
            input_image = get_file(
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
            )
            use_1k_labels = False
        else:
            input_image = get_file(
                "https://images.rawpixel.com/image_1300/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L3BkMTA2LTA0Ny1jaGltXzEuanBn.jpg"
            )
            use_1k_labels = True
        image = Image.open(str(input_image)).convert("RGB")
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        image = torch.rand(1, 3, 256, 256)
    pixel_values = transform(image).unsqueeze(0)

    inputs = [pixel_values.to(torch.bfloat16)]

    logger.info("framework_model={}", framework_model)

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
    fw_out, co_out = verify(inputs, framework_model, compiled_model)

    # Run model on sample data and print results
    print_cls_results(fw_out[0], co_out[0], use_1k_labels=use_1k_labels)
