# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest

import forge
import onnx
import torch
from forge.verify.verify import verify
import shutil
from forge.verify.config import VerifyConfig
from utils import load_inputs, load_model, print_results
from test.models.utils import Framework, Source, Task, build_module_name

variants = [
    "mobilenetv2_050",
    # "mobilenetv2_100",
    # "mobilenetv2_110d",
    # "mobilenetv2_140",
]


@pytest.mark.push
@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.nightly
def test_mobilenetv2_onnx(variant, forge_property_recorder):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.ONNX,
        model="mobilenetv2",
        variant=variant,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("priority")
    forge_property_recorder.record_model_name(module_name)

    # Load mobilenetv2 model
    onnx_model, onnx_dir_path = load_model(variant)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Load the inputs
    inputs = load_inputs(onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs, forge_property_handler=forge_property_recorder)

    pcc = 0.99
    if variant == "mobilenetv2_050":
        pcc = 0.98

    fw_out, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        forge_property_handler=forge_property_recorder,
        verify_cfg=VerifyConfig(pcc=pcc),
    )

    # Clean up the installation directory
    shutil.rmtree(onnx_dir_path)

    # Run model on sample data and print results
    print_results(fw_out[0], co_out[0])


