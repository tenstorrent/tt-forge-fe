# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from third_party.tt_forge_models.vit.pytorch import ModelLoader, ModelVariant

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

from test.models.models_utils import print_cls_results
from test.models.pytorch.vision.vision_utils.utils import load_vision_model_and_input

variants = [
    pytest.param(ModelVariant.BASE, marks=pytest.mark.push),
    ModelVariant.LARGE,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_vit_classify_224_hf_pytorch(variant):

    # Record Forge Property
    if variant == ModelVariant.BASE:
        group = ModelGroup.RED
        priority = ModelPriority.P1
    else:
        group = ModelGroup.GENERALITY
        priority = ModelPriority.P2

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.VIT,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
        group=group,
        priority=priority,
    )

    # Load model and inputs
    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)
    inputs = [inputs]

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
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing
    loader.post_processing(co_out)


variants_with_weights = {
    "vit_b_16": "ViT_B_16_Weights",
    "vit_b_32": "ViT_B_32_Weights",
    "vit_l_16": "ViT_L_16_Weights",
    "vit_l_32": "ViT_L_32_Weights",
    "vit_h_14": "ViT_H_14_Weights",
}

variants = [
    "vit_b_16",
    "vit_b_32",
    "vit_l_16",
    "vit_l_32",
    "vit_h_14",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_vit_torchvision(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.VIT,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCHVISION,
    )

    # Load model and input
    weight_name = variants_with_weights[variant]
    framework_model, inputs = load_vision_model_and_input(variant, "classification", weight_name)
    framework_model.to(torch.bfloat16)
    inputs = [inputs[0].to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    pcc = 0.99

    if variant in ["vit_b_32", "vit_l_32"]:
        pcc = 0.98
    elif variant == "vit_h_14":
        pcc = 0.93

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    fw_out, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )

    # Run model on sample data and print results
    print_cls_results(fw_out[0], co_out[0])
