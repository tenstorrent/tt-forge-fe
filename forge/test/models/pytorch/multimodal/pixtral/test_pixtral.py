# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    Source,
    Task,
    record_model_properties,
)

variants = ["mistralai/Pixtral-12B-2409", "mistralai/Pixtral-Large-Instruct-2411"]


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants)
def test_pixtral(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.PIXTRAL,
        variant=variant,
        task=Task.IMAGE_TEXT_PAIRING,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
    )

    raise RuntimeError("Requires multi-chip support")
