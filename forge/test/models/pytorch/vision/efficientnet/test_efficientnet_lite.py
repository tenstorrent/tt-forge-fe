# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
## https://github.com/RangiLyu/EfficientNet-Lite/
import pytest
import torch
from third_party.tt_forge_models.efficientnet_lite.pytorch import (
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
    ModelVariant.TF_EFFICIENTNET_LITE0_IN1K,
    ModelVariant.TF_EFFICIENTNET_LITE1_IN1K,
    ModelVariant.TF_EFFICIENTNET_LITE2_IN1K,
    ModelVariant.TF_EFFICIENTNET_LITE3_IN1K,
    ModelVariant.TF_EFFICIENTNET_LITE4_IN1K,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_efficientnet_lite_timm(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.EFFICIENTNETLITE,
        variant=variant.value,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load the model and inputs via loader
    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model(dtype_override=torch.bfloat16)
    input_tensor = loader.load_inputs(dtype_override=torch.bfloat16)
    inputs = [input_tensor]

    compiler_cfg = CompilerConfig(default_df_override=DataFormat.Float16_b)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    pcc = 0.99
    if variant in [ModelVariant.TF_EFFICIENTNET_LITE3_IN1K, ModelVariant.TF_EFFICIENTNET_LITE2_IN1K]:
        pcc = 0.98

    # Model Verification
    _, co_out = verify(
        inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc))
    )

    # Model Postprocessing
    loader.print_cls_results(co_out)
