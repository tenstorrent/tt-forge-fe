# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import torch

# from diffusers import StableDiffusionPipeline
# from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
import forge

from test.models.pytorch.multimodal.stable_diffusion.utils.model import (
    stable_diffusion_preprocessing,
    denoising_loop,
    stable_diffusion_postprocessing,
)


@pytest.mark.skip(reason="unsupported for now")
@pytest.mark.nightly
def test_stable_diffusion_pytorch(variant="CompVis/stable-diffusion-v1-4", batch_size=1):

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
    )

    # Data post-processing
    output = stable_diffusion_postprocessing(pipe, latents)
    output.images[0].save("/" + prompt.replace(" ", "_") + ".png")
