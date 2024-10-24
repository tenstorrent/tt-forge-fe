# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.model_demos.models.ghostnet import generate_model_ghostnet_imgcls_timm
import forge

variants = ["ghostnet_100"]


@pytest.mark.parametrize("variant", variants, ids=variants)
def test_ghostnet_timm(variant, test_device):
    model, inputs = generate_model_ghostnet_imgcls_timm(variant)
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=f"pt_{variant}")
