# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
import pytest
import torch
from third_party.tt_forge_models.yolov6.pytorch import ModelLoader, ModelVariant

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


class YoloV6Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        y, _ = self.model(x)
        # The model outputs float32, even if the input is bfloat16
        # Cast the output back to the input dtype
        return y.to(x.dtype)


# Didn't dealt with yolov6n6,yolov6s6,yolov6m6,yolov6l6 variants because of its higher input size(1280)
variants = [
    ModelVariant.YOLOV6N,
    ModelVariant.YOLOV6S,
    ModelVariant.YOLOV6M,
    ModelVariant.YOLOV6L,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_yolo_v6_pytorch(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.YOLOV6,
        variant=variant,
        source=Source.TORCH_HUB,
        task=Task.CV_OBJECT_DETECTION,
    )

    # Load model and inputs
    loader = ModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    framework_model = YoloV6Wrapper(model)
    input_tensor = loader.load_inputs(dtype_override=torch.bfloat16)
    inputs = [input_tensor]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, compiler_cfg=compiler_cfg
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model)
