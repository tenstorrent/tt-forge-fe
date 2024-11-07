# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import forge
from test.model_demos.models.wideresnet import (
    generate_model_wideresnet_imgcls_pytorch,
    generate_model_wideresnet_imgcls_timm,
)

variants = ["wide_resnet50_2", "wide_resnet101_2"]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.nightly
def test_wideresnet_pytorch(variant, test_device):
    (model, inputs,) = generate_model_wideresnet_imgcls_pytorch(
        test_device,
        variant,
    )

    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=f"pt_{variant}_hub")


variants = ["wide_resnet50_2", "wide_resnet101_2"]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.nightly
def test_wideresnet_timm(variant, test_device):
    (model, inputs,) = generate_model_wideresnet_imgcls_timm(
        test_device,
        variant,
    )

    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=f"pt_{variant}_timm")
