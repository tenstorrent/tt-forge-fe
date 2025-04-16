# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.pytorch.multimodal.stable_diffusion.utils.model import (
    load_pipe,
    stable_diffusion_preprocessing_xl,
)


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
@pytest.mark.skip_model_analysis
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "stable-diffusion-xl-base-1.0",
            marks=[pytest.mark.xfail],
        ),
    ],
)
def test_stable_diffusion_generation(forge_property_recorder, variant):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="stable_diffusion",
        variant=variant,
        task=Task.MUSIC_GENERATION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    forge_property_recorder.record_group("red")

    # Load the pipeline
    pipe = load_pipe(variant, variant_type="xl")

    # Extract only the unet, as the forward pass occurs here.
    framework_model = pipe.unet

    # Tokenize the prompt to a tensor
    tokenizer = pipe.tokenizer
    prompt = "An astronaut riding a green horse"
    (
        latent_model_input,
        timestep,
        prompt_embeds,
        timestep_cond,
        added_cond_kwargs,
        add_time_ids,
    ) = stable_diffusion_preprocessing_xl(pipe, prompt)
    inputs = [latent_model_input, timestep, prompt_embeds]

    # Wrap the pipeline in the wrapper
    framework_model = StableDiffusionXLWrapper(framework_model, added_cond_kwargs, cross_attention_kwargs=None)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
