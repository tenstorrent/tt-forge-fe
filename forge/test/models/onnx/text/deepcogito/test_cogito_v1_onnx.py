# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import onnx
from onnx import external_data_helper
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn

import forge
from forge.verify.verify import verify
from forge.forge_property_utils import Framework, Source, Task

from test.models.pytorch.text.deepcogito.utils.model import get_input_model


@pytest.mark.skip(reason="Skipping due to CI/CD Limitations")
@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["deepcogito/cogito-v1-preview-llama-3B"])
def test_cogito_generation_onnx(forge_property_recorder, tmp_path, variant):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.ONNX,
        model="cogito",
        variant=variant,
        task=Task.TEXT_GENERATION,
        source=Source.HUGGINGFACE,
    )
    forge_property_recorder.record_group("generality")

    # Load model and tokenizer
    sample_inputs, framework_model = get_input_model(variant)

    # Export paths
    temp_onnx = tmp_path / "temp_model.onnx"
    final_onnx = tmp_path / "cogito.onnx"
    external_data_file = tmp_path / "cogito.onnx.data"

    # Export to ONNX
    torch.onnx.export(
        framework_model,
        sample_inputs,
        str(temp_onnx),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch_size", 1: "seq_len"}},
        export_params=True,
        do_constant_folding=False,
    )

    onnx_model = onnx.load(str(temp_onnx))
    external_data_helper.convert_model_to_external_data(
        onnx_model,
        all_tensors_to_one_file=True,
        location=external_data_file.name,
        size_threshold=1024,
        convert_attribute=False,
    )
    onnx.save(onnx_model, str(final_onnx))

    # Load and validate ONNX model
    loaded_model = onnx.load(str(final_onnx), load_external_data=True)
    onnx.checker.check_model(loaded_model)

    # Create Forge ONNX model
    framework_model = forge.OnnxModule(module_name, loaded_model)

    # Compile with Forge
    compiled_model = forge.compile(
        loaded_model,
        sample_inputs=sample_inputs,
        module_name=module_name,
        forge_property_handler=forge_property_recorder,
    )

    # Run verification
    verify(
        sample_inputs,
        framework_model,
        compiled_model,
        forge_property_handler=forge_property_recorder,
    )
