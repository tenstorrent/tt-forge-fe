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


@pytest.mark.nightly
@pytest.mark.xfail
def test_uniad():

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.UNIAD,
        task=Task.PLANNING_ORIENTED_DRIVING,
        source=Source.GITHUB,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    raise RuntimeError("Test is currently not executable due to mmdet3d and model code dependency.")
