# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from third_party.tt_forge_models.stable_diffusion_xl.pytorch import (
    ModelLoader,
    ModelVariant,
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
from forge.verify.verify import verify


class StableDiffusionXLWrapper(torch.nn.Module):
    def __init__(self, model, added_cond_kwargs, cross_attention_kwargs=None):
        super().__init__()
        self.model = model
        self.cross_attention_kwargs = cross_attention_kwargs
        self.added_cond_kwargs = added_cond_kwargs

    def forward(self, latent_model_input, timestep, prompt_embeds):
        noise_pred = self.model(
            latent_model_input,
            timestep[0],
            encoder_hidden_states=prompt_embeds,
            timestep_cond=None,
            cross_attention_kwargs=self.cross_attention_kwargs,
            added_cond_kwargs=self.added_cond_kwargs,
        )[0]
        return noise_pred


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            ModelVariant.STABLE_DIFFUSION_XL_BASE_1_0,
            marks=[pytest.mark.xfail, pytest.mark.test_duration_check],
        ),
    ],
)
def test_stable_diffusion_generation(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.STABLEDIFFUSION,
        variant=variant,
        task=Task.CONDITIONAL_GENERATION,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    # Load model and input
    loader = ModelLoader(variant=variant)
    pipe = loader.load_model(dtype_override=torch.bfloat16)
    input_list = loader.load_inputs(dtype_override=torch.bfloat16)
    inputs = [input_list[0], input_list[1], input_list[2]]

    # Extract only the unet, as the forward pass occurs here.
    framework_model = pipe.unet

    # Wrap the pipeline in the wrapper
    framework_model = StableDiffusionXLWrapper(framework_model, input_list[3], cross_attention_kwargs=None)

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
    verify(inputs, framework_model, compiled_model)
