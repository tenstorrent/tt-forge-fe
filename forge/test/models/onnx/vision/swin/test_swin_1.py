# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import onnx
import torch
import forge
from test.models.pytorch.vision.utils.utils import load_vision_model_and_input
from forge.forge_property_utils import Framework, Source, Task

from torchvision.models.swin_transformer import ShiftedWindowAttentionV2

original_init = ShiftedWindowAttentionV2.__init__


def patched_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    print("Inside monkey-patched __init__ for ShiftedWindowAttentionV2")

    self.logit_scale = torch.tensor(torch.log(10 * torch.ones((self.num_heads, 1, 1))))


ShiftedWindowAttentionV2.__init__ = patched_init


variants_with_weights = {"swin_v2_t": "Swin_V2_T_Weights"}


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["swin_v2_t"])
def test_swin_torchvision(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.ONNX,
        model="swin",
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCHVISION,
    )
    forge_property_recorder.record_group("generality")

    # Load model and input
    weight_name = variants_with_weights[variant]
    framework_model, inputs = load_vision_model_and_input(variant, "classification", weight_name)

    # # Export model to ONNX
    onnx_path = "swin_v2_torchvision_f1.onnx"
    torch.onnx.export(
        framework_model,
        inputs[0],
        onnx_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        verbose=True,
    )

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(
        onnx_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )
