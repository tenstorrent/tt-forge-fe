# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from diffusers import StableDiffusionPipeline
from peft import OFTModel

import forge
from forge.verify.verify import verify

from test.models.pytorch.multimodal.oft.utils.oft_utils import (
    StableDiffusionWrapper,
    get_oft_configs,
    stable_diffusion_preprocessing,
)
from test.models.utils import Framework, Source, Task, build_module_name


@pytest.mark.xfail(
    reason="RuntimeError: Pivots given to lu_solve must all be greater or equal to 1. Did you properly pass the result of lu_factor?"
)
@pytest.mark.nightly
def test_oft_with_stable_diffusion_preprocessing(forge_property_recorder):

    # Build module name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="OFT",
        variant="with_sd_preprocessing",
        task=Task.CONDITIONAL_GENERATION,
        source=Source.GITHUB,
    )

    # Record Forge Property
    forge_property_recorder.record_group("priority")
    forge_property_recorder.record_model_name(module_name)

    # Load model
    model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    config_te, config_unet = get_oft_configs()
    model.text_encoder = OFTModel(model.text_encoder, config_te, "default")
    model.unet = OFTModel(model.unet, config_unet, "default")
    model.to("cpu")
    model.text_encoder.eval()
    model.unet.eval()

    # Load inputs
    prompt = "A beautiful mountain landscape during sunset"
    latents, timesteps, prompt_embeds, _ = stable_diffusion_preprocessing(
        model, prompt, device="cpu", num_inference_steps=1
    )
    sample_inputs = (latents, timesteps, prompt_embeds)

    # Forge compile framework model
    wrapped_model = StableDiffusionWrapper(model)
    compiled_model = forge.compile(
        wrapped_model,
        sample_inputs=sample_inputs,
        module_name=module_name,
        forge_property_recorder=forge_property_recorder,
    )

    # Model Verification
    verify(sample_inputs, wrapped_model, compiled_model, forge_property_recorder=forge_property_recorder)
