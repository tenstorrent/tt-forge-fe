# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import onnx

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.sam.utils.model import SamWrapper, get_model_inputs
from forge.forge_property_utils import Framework, Source, Task


@pytest.mark.xfail()
@pytest.mark.parametrize(
    "variant",
    [
        "facebook/sam-vit-huge",
        "facebook/sam-vit-large",
        "facebook/sam-vit-base",
    ],
)
@pytest.mark.nightly
def test_sam_onnx(forge_property_recorder, variant, tmp_path):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.ONNX,
        model="sam",
        variant=variant,
        task=Task.IMAGE_SEGMENTATION,
        source=Source.GITHUB,
    )

    if variant == "facebook/sam-vit-base":
        forge_property_recorder.record_group("red")
        forge_property_recorder.record_priority("P2")
    else:
        forge_property_recorder.record_group("generality")

    forge_property_recorder.record_model_name(module_name)

    # Load  model and input

    framework_model, sample_inputs = get_model_inputs(variant)
    input_tensor = sample_inputs[0]

    onnx_path = f"{tmp_path}/sam_" + str(variant).split("/")[-1].replace("-", "_") + ".onnx"
    torch.onnx.export(
        framework_model,
        input_tensor,
        onnx_path,
        input_names=["image"],
        output_names=["segmentation"],
        dynamic_axes={"image": {0: "batch_size"}},
    )

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=[input_tensor],
        module_name=module_name,
        forge_property_handler=forge_property_recorder,
    )

    # Model Verification
    verify(
        [input_tensor],
        framework_model,
        compiled_model,
        forge_property_handler=forge_property_recorder,
    )
