# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from hippynn.graphs import GraphModule, inputs, networks, targets


def load_model():
    """
    Load and initialize the Hippynn model.
    """
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
    network = networks.Hipnn("hippynn_model", (species, positions), module_kwargs=network_params)
    henergy = targets.HEnergyNode("HEnergy", network, db_name="T")

    # Load model
    framework_model = GraphModule([species, positions], [henergy.mol_energy])

    for param in framework_model.parameters():
        param.requires_grad = False

    return framework_model, henergy.mol_energy
