# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import ase.build
import ase.units
import pytest
import torch
from hippynn.graphs import inputs

import forge
from forge.verify.verify import verify

from test.models.pytorch.atomic.hippynn.utils.model import load_model
from test.models.utils import Framework, Source, Task, build_module_name

os.environ["HIPPYNN_USE_CUSTOM_KERNELS"] = "False"


class HippynWrapper(torch.nn.Module):
    def __init__(self, model, output_key):
        super().__init__()
        self.model = model
        self.output_key = output_key

    def forward(self, species: torch.Tensor, positions: torch.Tensor):
        input_dict = {"Z": species, "R": positions}
        output_dict = self.model(*input_dict.values())
        output_dict = list(output_dict)
        return output_dict


@pytest.mark.xfail
@pytest.mark.nightly
def test_hippynn(forge_property_recorder):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="Hippyn",
        variant="default",
        task=Task.ATOMIC_ML,
        source=Source.GITHUB,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # Load model
    framework_model, output_key = load_model()
    framework_model.eval()
    framework_model = HippynWrapper(framework_model, output_key=output_key)

    # Load inputs
    atoms = ase.build.molecule("H2O")
    pos = torch.as_tensor(atoms.positions / ase.units.Bohr).unsqueeze(0).to(torch.get_default_dtype())
    sp = torch.as_tensor(atoms.get_atomic_numbers()).unsqueeze(0)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=(sp, pos),
        module_name=module_name,
        forge_property_handler=forge_property_recorder,
    )
    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
