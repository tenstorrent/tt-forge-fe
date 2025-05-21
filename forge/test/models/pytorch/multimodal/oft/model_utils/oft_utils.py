# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import peft.tuners.oft.layer as oft_layer
import torch
from diffusers import StableDiffusionPipeline
from peft import OFTConfig, OFTModel


def get_oft_configs():
    config_te = OFTConfig(
        r=8,
        target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
        module_dropout=0.0,
        init_weights=True,
    )
    config_unet = OFTConfig(
        r=8,
        target_modules=[
            "proj_in",
            "proj_out",
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
        ],
        module_dropout=0.0,
        init_weights=True,
    )
    return config_te, config_unet


def patch_oft_cayley_with_lstsq():
    def _safe_cayley(self, data):
        data = data.detach()
        b, r, c = data.shape
        skew_mat = 0.5 * (data - data.transpose(1, 2))
        id_mat = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)
        try:
            return torch.linalg.solve(id_mat + skew_mat, id_mat - skew_mat)
        except RuntimeError:
            return torch.linalg.lstsq(id_mat + skew_mat, id_mat - skew_mat).solution

    oft_layer.Linear._cayley_batch = _safe_cayley


class StableDiffusionWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, latents, timesteps, prompt_embeds):
        return self.model.unet(
            latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
        ).sample


def get_inputs(model, prompt: str = "A beautiful mountain landscape during sunset", num_inference_steps: int = 30):
    patch_oft_cayley_with_lstsq()
    pipe = StableDiffusionPipeline.from_pretrained(model)
    config_te, config_unet = get_oft_configs()
    pipe.text_encoder = OFTModel(pipe.text_encoder, config_te, "default")
    pipe.unet = OFTModel(pipe.unet, config_unet, "default")

    for name, module in pipe.text_encoder.named_modules():
        if hasattr(module, "oft_r"):
            for key in module.oft_r:
                module.oft_r[key].requires_grad = False
        if hasattr(module, "oft_s"):
            for key in module.oft_r:
                module.oft_s[key].requires_grad = False

    for name, module in pipe.unet.named_modules():
        if hasattr(module, "oft_r"):
            for key in module.oft_r:
                module.oft_r[key].requires_grad = False
        if hasattr(module, "oft_s"):
            for key in module.oft_r:
                module.oft_s[key].requires_grad = False

    pipe.to("cpu")
    pipe.text_encoder.eval()
    pipe.unet.eval()
    prompt_embeds, negative_prompt_embeds, *_ = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt="",
        device="cpu",
        do_classifier_free_guidance=True,
        num_images_per_prompt=1,
    )
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    height = pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = height
    latents = torch.randn(
        (1, pipe.unet.config.in_channels, height // pipe.vae_scale_factor, width // pipe.vae_scale_factor)
    )
    latents = latents * pipe.scheduler.init_noise_sigma
    latents = torch.cat([latents] * 2, dim=0)

    pipe.scheduler.set_timesteps(num_inference_steps)
    timestep = pipe.scheduler.timesteps[0].expand(latents.shape[0])

    return pipe, (latents, timestep, prompt_embeds)
