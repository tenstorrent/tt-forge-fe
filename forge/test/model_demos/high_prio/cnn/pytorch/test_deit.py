# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import forge

from test.model_demos.models.deit import generate_model_deit_imgcls_hf_pytorch

variants = [
    "facebook/deit-base-patch16-224",
    "facebook/deit-base-distilled-patch16-224",
    "facebook/deit-small-patch16-224",
    "facebook/deit-tiny-patch16-224",
]


@pytest.mark.parametrize("variant", variants, ids=variants)
def test_vit_base_classify_224_hf_pytorch(variant, test_device):
    model, inputs, _ = generate_model_deit_imgcls_hf_pytorch(
        variant,
    )
    compiled_model = forge.compile(
        model, sample_inputs=inputs, module_name="pt_" + str(variant.split("/")[-1].replace("-", "_"))
    )
