# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from third_party.tt_forge_models.perceiverio_vision.pytorch.loader import (
    ModelLoader,
    ModelVariant,
)

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
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

variants = [
    ModelVariant.VISION_PERCEIVER_CONV,
    pytest.param(
        ModelVariant.VISION_PERCEIVER_LEARNED,
        marks=pytest.mark.xfail,
    ),
    pytest.param(
        ModelVariant.VISION_PERCEIVER_FOURIER,
        marks=pytest.mark.xfail,
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_perceiverio_for_image_classification_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.PERCEIVERIO,
        variant=variant,
        task=Task.CV_IMAGE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load model and inputs using ModelLoader
    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model()
    framework_model.to(torch.bfloat16)

    pixel_values = loader.load_inputs(dtype_override=torch.bfloat16)
    inputs = [pixel_values]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    _, co_out = verify(
        inputs, framework_model, compiled_model, verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.98))
    )

    # Run model on sample data and print results
    loader.print_cls_results(co_out)
