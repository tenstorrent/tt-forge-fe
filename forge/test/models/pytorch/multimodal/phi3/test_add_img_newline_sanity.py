# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn

import forge
from forge.verify.verify import DepricatedVerifyConfig, verify


def test_embedding_sanity(forge_property_recorder):
    class AddImageNewline(nn.Module):
        def __init__(self, image_dim_out=1024):
            super().__init__()
            self.image_dim_out = image_dim_out
            self.sub_GN = torch.zeros(1, 1, 1, image_dim_out * 4)

        def forward(self, image_features_hd):
            """
            image_features_hd: (num_images, h_crop*12, w_crop*12, 4096)
            output: (num_images, (h_crop*12) * (w_crop*12 + 1), 4096)
            """
            num_images, h, w, hid_dim = image_features_hd.shape
            newline_embeddings = self.sub_GN.expand(num_images, h, 1, hid_dim)
            image_features_hd_newline = torch.cat([image_features_hd, newline_embeddings], dim=2)
            image_features_hd_newline = image_features_hd_newline.reshape(num_images, -1, hid_dim)
            return image_features_hd_newline

    module = AddImageNewline()
    input_tensor = torch.randn(1, 12, 12, 4096)
    # Forge compile framework model
    compiled_model = forge.compile(
        module,
        sample_inputs=[input_tensor],
        module_name="AddImageNewline",
        verify_cfg=DepricatedVerifyConfig(verify_forge_codegen_vs_framework=True),
        forge_property_handler=forge_property_recorder,
    )

    # Model Verification
    verify([input_tensor], module, compiled_model, forge_property_handler=forge_property_recorder)
