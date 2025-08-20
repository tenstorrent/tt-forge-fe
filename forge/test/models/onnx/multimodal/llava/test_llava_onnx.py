# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import onnx

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    ModelPriority,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify
from third_party.tt_forge_models.llava.pytorch import (
    ModelLoader as ConditionalGenModelLoader,
)
from third_party.tt_forge_models.llava.pytorch import (
    ModelVariant as ConditionalGenModelVariant,
)


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, pixel_values):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
        output = self.model(**inputs)
        return output.logits


LLAVA_VARIANTS = [
    ConditionalGenModelVariant.LLAVA_1_5_7B,
]


@pytest.mark.xfail
@pytest.mark.nightly
@pytest.mark.parametrize("variant", LLAVA_VARIANTS)
def test_llava_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.LLAVA,
        variant=variant,
        task=Task.CONDITIONAL_GENERATION,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    pytest.xfail(reason="https://github.com/tenstorrent/tt-forge-fe/issues/2832")
    # Load model and inputs
    loader = ConditionalGenModelLoader()
    torch_model = loader.load_model()
    wrapped_model = Wrapper(torch_model)

    inputs_dict = loader.load_inputs()
    input_ids = inputs_dict["input_ids"]
    attention_mask = inputs_dict["attention_mask"]
    pixel_values = inputs_dict["pixel_values"]

    inputs = [input_ids, attention_mask, pixel_values]

    # ONNX export path
    onnx_path = f"{forge_tmp_path}/{str(variant).lower().replace('-', '_')}.onnx"

    # Export to ONNX
    torch.onnx.export(
        wrapped_model,
        (input_ids, attention_mask, pixel_values),
        onnx_path,
        input_names=["input_ids", "attention_mask", "pixel_values"],
    )

    # Load and validate ONNX model
    onnx_model = onnx.load(onnx_path)

    framework_model = forge.OnnxModule(module_name, onnx_model, onnx_path)

    # Compile ONNX model
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
