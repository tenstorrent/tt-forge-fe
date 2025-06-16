# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.pytorch.multimodal.stable_diffusion.model_utils.model import (
    load_pipe,
    stable_diffusion_preprocessing_v35,
)


class StableDiffusionWrapper(torch.nn.Module):
    def __init__(self, model, joint_attention_kwargs=None, return_dict=False):
        super().__init__()
        self.model = model
        self.joint_attention_kwargs = joint_attention_kwargs
        self.return_dict = return_dict

    def forward(self, latent_model_input, timestep, prompt_embeds, pooled_prompt_embeds):
        noise_pred = self.model(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=self.return_dict,
        )[0]
        return noise_pred


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "stable-diffusion-3.5-medium",
        ),
        pytest.param(
            "stable-diffusion-3.5-large",
        ),
        pytest.param(
            "stable-diffusion-3.5-large-turbo",
        ),
    ],
)
def test_stable_diffusion_v35(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.STABLEDIFFUSION,
        variant=variant,
        task=Task.CONDITIONAL_GENERATION,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
    )

    raise RuntimeError("Requires multi-chip support")

    # Load pipeline
    pipe = load_pipe(variant, variant_type="v35")

    # Extract only the transformer, as the forward pass occurs here.
    framework_model = pipe.transformer

    # TODO: Implement post-processing using VAE decode after obtaining the transformer output.
    framework_model = StableDiffusionWrapper(framework_model, joint_attention_kwargs=None, return_dict=False)
    framework_model.to(torch.bfloat16)

    # Load inputs
    prompt = "An astronaut riding a green horse"
    latent_model_input, timestep, prompt_embeds, pooled_prompt_embeds = stable_diffusion_preprocessing_v35(pipe, prompt)
    inputs = [
        latent_model_input.to(torch.bfloat16),
        timestep.to(torch.bfloat16),
        prompt_embeds.to(torch.bfloat16),
        pooled_prompt_embeds.to(torch.bfloat16),
    ]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, compiler_cfg=compiler_cfg
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model)
