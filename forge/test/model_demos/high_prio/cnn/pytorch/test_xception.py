# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import forge
from test.model_demos.models.xception import generate_model_xception_imgcls_timm

variants = ["xception", "xception41", "xception65", "xception71"]


@pytest.mark.parametrize("variant", variants, ids=variants)
def test_xception_timm(variant, test_device):

    (model, inputs,) = generate_model_xception_imgcls_timm(
        test_device,
        variant,
    )
    compiled_model = forge.compile(model, sample_inputs=inputs)
