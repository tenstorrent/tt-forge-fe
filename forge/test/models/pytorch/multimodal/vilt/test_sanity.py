# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
from transformers import ViltConfig, ViltForQuestionAnswering

import forge

from test.utils import download_model


@pytest.mark.parametrize("compile_time_shapes", [True, False])
def test_vilt(compile_time_shapes):
    class emb(nn.Module):
        def __init__(self, model):
            super().__init__()

            num_patches = 144
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
            self.config = model.config

        def visual_embed(self, x_mask):

            x_h = x_mask[:, 0].sum(dim=1)[:, 0]
            x_w = x_mask[:, 0].sum(dim=2)[:, 0]

            batch_size, num_channels, height, width = (1, 768, 12, 16)
            patch_dim = self.config.image_size // self.config.patch_size
            spatial_pos = self.position_embeddings[:, 1:, :].transpose(1, 2).view(1, num_channels, patch_dim, patch_dim)

            if compile_time_shapes:
                # custom implementation for check
                h = 12
                w = 16
                resized_pos = nn.functional.interpolate(spatial_pos, size=(h, w), mode="bilinear", align_corners=True)
            else:
                # org implementation - https://github.com/huggingface/transformers/blob/5d7739f15a6e50de416977fe2cc9cb516d67edda/src/transformers/models/vilt/modeling_vilt.py#L125
                for h, w in zip(x_h, x_w):
                    resized_pos = nn.functional.interpolate(
                        spatial_pos, size=(h, w), mode="bilinear", align_corners=True
                    )
                    break

            return resized_pos

        def forward(self, pixel_mask):
            image_embeds = self.visual_embed(pixel_mask)
            return image_embeds

    variant = "dandelin/vilt-b32-finetuned-vqa"
    config = ViltConfig.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config = ViltConfig(**config_dict)

    m = download_model(ViltForQuestionAnswering.from_pretrained, variant, config=config)
    m.eval()

    model = emb(m)
    model.eval()

    x_mask = torch.load("x_mask.pt")

    inputs = [x_mask]

    compiled_model = forge.compile(model, sample_inputs=inputs, module_name="jan26_vilt_s")
