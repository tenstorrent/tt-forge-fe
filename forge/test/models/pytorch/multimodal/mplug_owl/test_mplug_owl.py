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

variants = ["mPLUG/mPLUG-Owl3-7B-240728"]


@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
@pytest.mark.xfail
def test_mplug_owl(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.MPLUGOWL,
        variant=variant,
        task=Task.MULTIMODAL_TEXT_GENERATION,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
    )

    raise RuntimeError("Requires multi-chip support")
