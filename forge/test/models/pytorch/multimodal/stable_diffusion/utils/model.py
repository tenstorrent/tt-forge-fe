# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Stable Diffusion Demo Script

from typing import List, Optional, Union

import torch

# from diffusers import StableDiffusionPipeline
# from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
import forge
from test.models.utils import build_module_name


def stable_diffusion_preprocessing(
    pipeline,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    callback_steps: int = 1,
):

    # 0. Default height and width to unet
    height = height or pipeline.unet.config.sample_size * pipeline.vae_scale_factor

    print("pipeline.unet.config.sample_size", pipeline.unet.config.sample_size)
    print("pipeline.vae_scale_factor", pipeline.vae_scale_factor)
    print("width", width)

    width = width or pipeline.unet.config.sample_size * pipeline.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    pipeline.check_inputs(
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
    )

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = pipeline._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    prompt_embeds = pipeline._encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )

    # 4. Prepare timesteps
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipeline.scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = pipeline.unet.in_channels
    latents = pipeline.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta)

    return (
        latents,
        timesteps,
        extra_step_kwargs,
        prompt_embeds,
        extra_step_kwargs,
    )


class UnetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, latent_model_input, timestep, text_embeddings):
        noise_pred = self.unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings)["sample"]
        return noise_pred


def initialize_compiler_overrides():

    compiler_cfg = forge.config._get_global_compiler_config()


def denoising_loop(
    pipeline,
    latents,
    timesteps,
    prompt_embeds,
    extra_step_kwargs,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    callback=None,
    callback_steps: int = 1,
):

    do_classifier_free_guidance = guidance_scale > 1.0
    num_warmup_steps = len(timesteps) - num_inference_steps * pipeline.scheduler.order
    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            if do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
                timestep_ = torch.cat([t.unsqueeze(0)] * 2).float()
            else:
                latent_model_input = latents
                timestep_ = t

            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

            # sanity
            # noise_pred_0 = pipeline(latent_model_input.detach()[0:1],timestep_.detach()[0:1],prompt_embeds.detach()[0:1],)

            inputs = [latent_model_input.detach()[0:1], timestep_.detach()[0:1], prompt_embeds.detach()[0:1]]
            module_name = build_module_name(framework="pt", model="stable_diffusion", suffix=f"1_{i}")
            compiled_model = forge.compile(pipeline, sample_inputs=inputs, module_name=module_name)
            noise_pred_0 = compiled_model(*inputs)

            # sanity
            # noise_pred_1 = pipeline(latent_model_input.detach()[1:2],timestep_.detach()[1:2],prompt_embeds.detach()[1:2],)
            inputs = [latent_model_input.detach()[1:2], timestep_.detach()[1:2], prompt_embeds.detach()[1:2]]
            module_name = build_module_name(framework="pt", model="stable_diffusion", suffix=f"2_{i}")
            compiled_model = forge.compile(pipeline, sample_inputs=inputs, module_name=module_name)
            noise_pred_1 = compiled_model(*inputs)

            noise_pred = torch.cat([noise_pred_0[0].value().detach(), noise_pred_1[0].value().detach()], dim=0)

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = pipeline.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    return latents


def stable_diffusion_postprocessing(
    pipeline,
    latents,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
):

    has_nsfw_concept = None
    if output_type == "latent":
        image = latents
        has_nsfw_concept = None
    elif output_type == "pil":
        # 8. Post-processing
        latents = 1 / pipeline.vae.config.scaling_factor * latents
        image = pipeline.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().detach().numpy()

        # 9. Convert to PIL
        image = pipeline.numpy_to_pil(image)
    else:
        # 8. Post-processing
        latents = 1 / pipeline.vae.config.scaling_factor * latents
        image = pipeline.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().detach().numpy()

    # Offload last model to CPU
    if hasattr(pipeline, "final_offload_hook") and pipeline.final_offload_hook is not None:
        pipeline.final_offload_hook.offload()

    if not return_dict:
        return (image, has_nsfw_concept)

    return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
