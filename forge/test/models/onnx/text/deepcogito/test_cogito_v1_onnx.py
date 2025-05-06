# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import onnx

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.pytorch.text.deepcogito.utils.model import get_input_model


# @pytest.mark.skip("RuntimeError: The serialized model is larger than the 2GiB limit imposed by the protobuf library.")
@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["deepcogito/cogito-v1-preview-llama-3B"])
def test_cogito_generation_onnx(forge_property_recorder, variant, tmp_path):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.ONNX,
        model="cogito",
        variant=variant,
        task=Task.TEXT_GENERATION,
        source=Source.HUGGINGFACE,
    )
    forge_property_recorder.record_group("generality")

    # Load PyTorch model and inputs
    input_tensor_list, model = get_input_model(variant)
    input_tensor = input_tensor_list[0]

    # Export to ONNX
    onnx_path = str(tmp_path / "cogito.onnx")

    torch.onnx.export(
        model,
        input_tensor,
        onnx_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch_size", 1: "seq_len"}},
        export_params=True,
    )

    onnx_model = onnx.load(onnx_path)
    onnx.save_model(onnx_model, onnx_path, save_as_external_data=True)

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    framework_model = forge.OnnxModule(module_name, onnx_model, onnx_path)

    # Compile and verify
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=[input_tensor],
        module_name=module_name,
        forge_property_handler=forge_property_recorder,
    )

    verify([input_tensor], framework_model, compiled_model, forge_property_handler=forge_property_recorder)
