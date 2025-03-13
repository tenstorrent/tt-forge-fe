# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import ase.build
import ase.units
import pytest
import torch
from hippynn.graphs import GraphModule, inputs, networks, targets

import forge
from forge.verify.verify import verify

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
        return output_dict[0] if isinstance(output_dict, tuple) else output_dict


@pytest.mark.xfail(reason="Exception: warning unhandled case: <class 'tvm.relay.expr.TupleWrapper'>")
@pytest.mark.nightly
def test_hippynn(record_forge_property):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="Hippyn",
        variant="default",
        task=Task.ATOMIC_ML,
        source=Source.GITHUB,
    )

    # Record Forge Property
    record_forge_property("group", "priority")
    record_forge_property("tags.model_name", module_name)
    torch.set_default_dtype(torch.float64)

    network_params = {
        "possible_species": [0, 1, 6, 7, 8, 16],
        "n_features": 20,
        "n_sensitivities": 20,
        "dist_soft_min": 1.6,
        "dist_soft_max": 10.0,
        "dist_hard_max": 12.5,
        "n_interaction_layers": 2,
        "n_atom_layers": 3,
    }

    species = inputs.SpeciesNode(db_name="Z")
    positions = inputs.PositionsNode(db_name="R")
    network = networks.Hipnn("hipnn_model", (species, positions), module_kwargs=network_params)
    henergy = targets.HEnergyNode("HEnergy", network, db_name="T")

    # Load model
    framework_model = GraphModule([species, positions], [henergy.mol_energy])
    framework_model.eval()

    for param in framework_model.parameters():
        param.requires_grad = False
        if param.dtype in (torch.float32, torch.float64):
            torch.nn.init.normal_(param, mean=0.0, std=0.1)

    # Load inputs
    atoms = ase.build.molecule("H2O")
    pos = torch.as_tensor(atoms.positions / ase.units.Bohr).unsqueeze(0).to(torch.get_default_dtype())
    sp = torch.as_tensor(atoms.get_atomic_numbers()).unsqueeze(0)

    framework_model = HippynWrapper(framework_model, output_key=henergy.mol_energy)
    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=(sp, pos), module_name=module_name)
    # Model Verification
    verify(inputs, framework_model, compiled_model)
