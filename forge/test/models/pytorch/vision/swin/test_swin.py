# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# STEP 0: import Forge library
import pytest
import torch
from third_party.tt_forge_models.swin.image_classification.pytorch import (
    ModelLoader,
    ModelVariant,
)
from third_party.tt_forge_models.swin.masked_image_modeling.pytorch import (
    ModelLoader as MaskedImageModelingLoader,
)
from third_party.tt_forge_models.swin.masked_image_modeling.pytorch import (
    ModelVariant as MaskedImageModelingVariant,
)

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    ModelPriority,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

variants = [
    pytest.param(
        ModelVariant.SWIN_TINY_HF,
        marks=pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-forge-fe/issues/2998"),
    ),
    pytest.param(ModelVariant.SWINV2_TINY_HF, marks=pytest.mark.xfail),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_swin_hf_image_classification(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.SWIN,
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.IMAGE_CLASSIFICATION,
    )

    dtype_override = None
    if variant == ModelVariant.SWIN_TINY_HF:
        dtype_override = torch.bfloat16
        data_format_override = DataFormat.Float16_b
        compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    elif variant == ModelVariant.SWINV2_TINY_HF:
        compiler_cfg = CompilerConfig()

    # Load model and inputs
    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model(dtype_override=dtype_override)
    framework_model.config.return_dict = False
    input_tensor = loader.load_inputs(dtype_override=dtype_override)
    inputs = [input_tensor]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    pcc = 0.99
    if variant == ModelVariant.SWIN_TINY_HF:
        pcc = 0.95

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )


@pytest.mark.nightly
@pytest.mark.skip_model_analysis
@pytest.mark.xfail
@pytest.mark.parametrize("variant", [MaskedImageModelingVariant.SWINV2_TINY])
def test_swin_v2_tiny_masked(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.SWIN,
        variant=variant,
        task=Task.MASKED_IMAGE_MODELING,
        source=Source.HUGGINGFACE,
    )

    # Load model and inputs
    loader = MaskedImageModelingLoader(variant=variant)
    framework_model = loader.load_model()
    framework_model.config.return_dict = False
    input_tensor = loader.load_inputs()
    inputs = [input_tensor]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


variants = [
    ModelVariant.SWIN_T,
    ModelVariant.SWIN_S,
    ModelVariant.SWIN_B,
    pytest.param(ModelVariant.SWIN_V2_T),
    pytest.param(ModelVariant.SWIN_V2_S),
    pytest.param(ModelVariant.SWIN_V2_B),
]


@pytest.mark.nightly
@pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-forge-fe/issues/2998")
@pytest.mark.parametrize("variant", variants)
def test_swin_torchvision(variant):

    if variant == ModelVariant.SWIN_V2_S:
        group = ModelGroup.RED
        priority = ModelPriority.P1
    else:
        group = ModelGroup.GENERALITY
        priority = ModelPriority.P2

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.SWIN,
        variant=variant.value,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCHVISION,
        group=group,
        priority=priority,
    )

    # Load model and input using ModelLoader
    loader = ModelLoader(variant=variant)

    if variant in [ModelVariant.SWIN_T, ModelVariant.SWIN_S, ModelVariant.SWIN_B]:
        # Load model and inputs with bfloat16 for Swin v1 variants
        framework_model = loader.load_model(dtype_override=torch.bfloat16)
        inputs = loader.load_inputs(dtype_override=torch.bfloat16)
        inputs = [inputs]  # Wrap in list to match expected format

        data_format_override = DataFormat.Float16_b
        compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    else:
        # Load model and inputs with default dtype for Swin v2 variants
        framework_model = loader.load_model()
        inputs = loader.load_inputs()
        inputs = [inputs]  # Wrap in list to match expected format

        compiler_cfg = CompilerConfig()

    pcc = 0.99

    if variant in [ModelVariant.SWIN_T, ModelVariant.SWIN_V2_T, ModelVariant.SWIN_V2_S]:
        pcc = 0.95
    elif variant == ModelVariant.SWIN_S:
        pcc = 0.92
    elif variant == ModelVariant.SWIN_B:
        pcc = 0.93

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )

    # Post processing
    loader.print_cls_results(co_out)
