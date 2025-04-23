# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.sam.utils.model import SamWrapper, get_model_inputs
from test.models.utils import Framework, Source, Task, build_module_name


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
def test_sam_onnx(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="sam",
        variant=variant,
        task=Task.IMAGE_SEGMENTATION,
        source=Source.GITHUB,
    )

    if variant == "facebook/sam-vit-base":
        forge_property_recorder.record_group("red")
        forge_property_recordeer.record_priority("P2")
    else:
        forge_property_recorder.record_group("generality")

    forge_property_recorder.record_model_name(module_name)

    # Load  model and input

    framework_model, sample_inputs = get_model_inputs(variant)

    input_tensor = sample_inputs[0]
    onnx_path_str = str(tmp_path / "sam.onnx")

    # Export to ONNX
    torch.onnx.export(
        model,
        input_tensor,
        onnx_path_str,
        input_names=["image"],
        output_names=["segmentation"],
        dynamic_axes={"image": {0: "batch_size"}},
        opset_version=17,
    )

    # Save model with external tensor data if necessary
    onnx_model = onnx.load(onnx_path_str)
    onnx.save_model(onnx_model, onnx_path_str, save_as_external_data=True)

    # Forge ONNX inference
    framework_model = forge.OnnxModule(module_name, onnx_model, onnx_path_str)
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=[input_tensor],
        module_name=module_name,
        forge_property_handler=forge_property_recorder,
    )

    verify([input_tensor], framework_model, compiled_model, forge_property_handler=forge_property_recorder)
