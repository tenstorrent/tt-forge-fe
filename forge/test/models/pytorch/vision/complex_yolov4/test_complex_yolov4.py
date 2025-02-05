# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from easydict import EasyDict as edict

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.complex_yolov4.utils.model_utils import create_model
from test.models.utils import Framework, Source, Task, build_module_name


@pytest.mark.parametrize("variant", ["complex_yolov4_tiny", "complex_yolov4"])
def test_compelx_yolov4(record_forge_property, variant):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="complex_yolov4",
        variant=variant,
        source=Source.GITHUB,
        task=Task.OBJECT_DETECTION_3D,
    )

    # Record Forge Property
    record_forge_property("model_name", module_name)

    # Load model
    configs = edict(
        {
            "arch": "darknet",
            "cfgfile": f"forge/test/models/pytorch/vision/complex_yolov4/utils/{variant}.cfg",
        }
    )
    model = create_model(configs)
    model.eval()

    # prepare sample input
    inputs = [torch.randn((1, 3, 608, 608))]

    # Forge compile framework model
    compiled_model = forge.compile(model, inputs, module_name=module_name)

    # Model Verification
    verify(inputs, model, compiled_model)
