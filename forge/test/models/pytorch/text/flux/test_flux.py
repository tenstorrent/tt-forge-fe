# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    ModelPriority,
    Source,
    Task,
    record_model_properties,
)

variants = ["black-forest-labs/FLUX.1-schnell", "black-forest-labs/FLUX.1-dev"]


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants)
def test_flux(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.FLUX,
        variant=variant,
        task=Task.CONDITIONAL_GENERATION,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    pytest.xfail(reason="Requires multi-chip support")
