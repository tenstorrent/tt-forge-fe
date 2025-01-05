# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

from test.models.pytorch.multimodal.stable_diffusion.utils.model import (
    stable_diffusion_preprocessing,
    denoising_loop,
    stable_diffusion_postprocessing,
)
from test.models.utils import build_module_name, Framework


@pytest.mark.skip(reason="unsupported for now")
@pytest.mark.nightly
def test_stable_diffusion_pytorch(record_forge_property):
    variant = "CompVis/stable-diffusion-v1-4"

    module_name = build_module_name(framework=Framework.PYTORCH, model="stable_diffusion", variant=variant)

    record_forge_property("module_name", module_name)

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
    )

    # Data post-processing
    output = stable_diffusion_postprocessing(pipe, latents)
    output.images[0].save("/" + prompt.replace(" ", "_") + ".png")
