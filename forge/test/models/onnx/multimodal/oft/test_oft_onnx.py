# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

import forge
from forge.verify.verify import verify

from test.models.onnx.multimodal.oft.model_utils.oft_utils import get_inputs, get_models
from forge.forge_property_utils import Framework, Source, Task, ModelPriority, ModelArch, record_model_properties


@pytest.mark.skip(reason="Segmentation Fault")
@pytest.mark.parametrize("variant", ["runwayml/stable-diffusion-v1-5"])
@pytest.mark.nightly
def test_oft(forge_tmp_path, variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.OFT,
        variant=variant.split("/")[-1],
        task=Task.CONDITIONAL_GENERATION,
        source=Source.HUGGINGFACE,
        priority=ModelPriority.P1,
    )

    # Load model and inputs
    pipe, inputs = get_inputs(model=variant)
    onnx_model, framework_model = get_models(inputs, forge_tmp_path, pipe)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=inputs,
        module_name=module_name,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model)
