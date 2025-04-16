# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

from forge.forge_property_utils import Framework

from test.models.pytorch.multimodal.stable_diffusion.utils.model import (
    denoising_loop,
    stable_diffusion_postprocessing,
    stable_diffusion_preprocessing,
)


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="unsupported for now")
@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["CompVis/stable-diffusion-v1-4"])
def test_stable_diffusion_pytorch(forge_property_recorder, variant):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH, model="stable_diffusion", variant=variant
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    batch_size = 1

    # Set inference steps
    num_inference_steps = 50

    # Load model
    pipe = StableDiffusionPipeline.from_pretrained(variant)

    # Sample prompt
    prompt = "An image of a cat"
    print("Generating image for prompt: ", prompt)

    # Data preprocessing
    (latents, timesteps, extra_step_kwargs, prompt_embeds, extra_step_kwargs,) = stable_diffusion_preprocessing(
        pipe,
        [prompt] * batch_size,
        num_inference_steps=num_inference_steps,
    )

    # Run inference
    latents = denoising_loop(
        pipe,
        latents,
        timesteps,
        prompt_embeds,
        extra_step_kwargs,
        num_inference_steps=num_inference_steps,
        forge_property_handler=forge_property_recorder,
    )

    # Data post-processing
    output = stable_diffusion_postprocessing(pipe, latents)
    output.images[0].save("/" + prompt.replace(" ", "_") + ".png")
