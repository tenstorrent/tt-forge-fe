# # SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# # SPDX-License-Identifier: Apache-2.0

import sys

sys.path.append("forge/test/models/pytorch/multimodal/trajectron/trajectron/")
import pytest
import forge
from test.models.pytorch.multimodal.trajectron.trajectron.model import Trajectron
from test.models.pytorch.multimodal.trajectron.trajectron.model.model_registrar import ModelRegistrar
from test.models.pytorch.multimodal.trajectron.trajectron.model.dataset import (
    EnvironmentDataset,
    collate,
    get_timesteps_data,
)
from forge.verify.compare import compare_with_golden
import os
import json
import dill
import torch
import torch.nn as nn
import numpy as np
from typing import Any
import torch.nn.utils.rnn as rnn
import pytest


def load_hyperparams():
    conf_path = "forge/test/models/pytorch/multimodal/trajectron/trajectron/config/config.json"
    with open(conf_path, "r", encoding="utf-8") as conf_json:
        hyperparams = json.load(conf_json)

    # Set Default values
    hyperparams["scene_freq_mult_eval"] = False
    hyperparams["node_freq_mult_eval"] = False
    hyperparams["edge_encoding"] = False
    hyperparams["incl_robot_node"] = False
    hyperparams["use_map_encoding"] = False

    hyperparams["edge_addition_filter"] = [1, 1]
    hyperparams["edge_removal_filter"] = [1, 1]

    return hyperparams


def load_env():
    eval_data_path = "forge/test/models/pytorch/multimodal/trajectron/trajectron/dataset_envs/eth_val.pkl"
    with open(eval_data_path, "rb") as f:
        eval_env = dill.load(f, encoding="latin1")
    return eval_env


class TrajectronWrapper(nn.Module):
    def __init__(
        self,
        model_dir: str,
        hyperparams: dict[str, Any],
        env: Any,
        scene_index: int,
        num_samples: int = 1,
        z_mode: bool = True,
        gmm_mode: bool = True,
        all_z_sep: bool = False,
        full_dist: bool = False,
    ):
        super().__init__()

        # Build Model registrar
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=False)
        model_config_path = model_dir + "/config.json"
        if not os.path.exists(model_config_path):
            with open(model_config_path, "w") as conf_json:
                json.dump(hyperparams, conf_json)
        model_registrar = ModelRegistrar(model_dir, "cpu")

        # Build Trajectron Model
        self.model = Trajectron(model_registrar=model_registrar, hyperparams=hyperparams, log_writer=None, device="cpu")
        self.model.set_environment(env=env)

        self.model_dir = model_dir
        self.hyperparams = hyperparams
        self.env = env

        assert len(self.env.NodeType) == 1
        self.node_type = self.env.NodeType[0]

        self.scene_index = scene_index
        self.num_samples = num_samples
        self.z_mode = z_mode
        self.gmm_mode = gmm_mode
        self.all_z_sep = all_z_sep
        self.full_dist = full_dist

    def _build_packed_sequence(
        self,
        packed_sequence_data,
        packed_sequence_batch_sizes,
        packed_sequence_sorted_indices,
        packed_sequence_unsorted_indices,
    ):
        packed_sequence = torch.nn.utils.rnn.PackedSequence(
            data=packed_sequence_data.squeeze(),
            batch_sizes=packed_sequence_batch_sizes.squeeze(),
            sorted_indices=packed_sequence_sorted_indices.squeeze(),
            unsorted_indices=packed_sequence_unsorted_indices.squeeze(),
        )
        return packed_sequence

    def forward(
        self,
        x,
        x_st_t,
        packed_sequence_data,
        packed_sequence_batch_sizes,
        packed_sequence_sorted_indices,
        packed_sequence_unsorted_indices,
        first_history_index,
    ):
        neighbors_data_st = None
        neighbors_edge_value = None
        robot_traj_st_t = None
        map = None

        ph = self.hyperparams["prediction_horizon"]

        packed_x_st_t = self._build_packed_sequence(
            packed_sequence_data,
            packed_sequence_batch_sizes,
            packed_sequence_sorted_indices,
            packed_sequence_unsorted_indices,
        )

        model = self.model.node_models_dict[self.node_type]
        predictions = model.predict(
            inputs=x,
            inputs_st=x_st_t,  # Pack and send this
            packed_inputs_st=packed_x_st_t,
            first_history_indices=first_history_index,
            neighbors=neighbors_data_st,
            neighbors_edge_value=neighbors_edge_value,
            robot=robot_traj_st_t,
            map=map,
            prediction_horizon=ph,
            num_samples=self.num_samples,
            z_mode=self.z_mode,
            gmm_mode=self.gmm_mode,
            full_dist=self.full_dist,
            all_z_sep=self.all_z_sep,
        )

        return predictions

    def eval(self):
        super().eval()
        self.model.eval()

    def get_input_batch(self, scene):
        ph = self.hyperparams["prediction_horizon"]
        timesteps = scene.sample_timesteps(1, min_future_timesteps=ph)

        min_future_timesteps = ph
        min_history_timesteps = 1

        node_type = self.node_type
        assert node_type in self.model.pred_state
        model = self.model.node_models_dict[node_type]

        # Get Input data for node type and given timesteps
        batch = get_timesteps_data(
            env=self.env,
            scene=scene,
            t=timesteps,
            node_type=node_type,
            state=self.model.state,
            pred_state=self.model.pred_state,
            edge_types=model.edge_types,
            min_ht=min_history_timesteps,
            max_ht=self.model.max_ht,
            min_ft=min_future_timesteps,
            max_ft=min_future_timesteps,
            hyperparams=self.hyperparams,
        )

        assert batch is not None

        (
            (
                first_history_index,
                x_t,
                y_t,
                x_st_t,
                y_st_t,
                neighbors_data_st,
                neighbors_edge_value,
                robot_traj_st_t,
                map,
            ),
            nodes,
            timesteps_o,
        ) = batch

        device = self.model.device
        x = x_t.to(device)
        x_st_t = x_st_t.to(device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(device)

        if type(map) == torch.Tensor:
            map = map.to(device)

        return (x, x_st_t, first_history_index, neighbors_data_st, neighbors_edge_value, robot_traj_st_t, map), (
            nodes,
            timesteps_o,
        )


def pack_input_sequences(sequences, lower_indices=None, upper_indices=None, total_length=None):
    bs, tf = sequences.shape[:2]
    if lower_indices is None:
        lower_indices = torch.zeros(bs, dtype=torch.int)
    if upper_indices is None:
        upper_indices = torch.ones(bs, dtype=torch.int) * (tf - 1)
    if total_length is None:
        total_length = max(upper_indices) + 1
    # This is done so that we can just pass in self.prediction_timesteps
    # (which we want to INCLUDE, so this will exclude the next timestep).
    inclusive_break_indices = upper_indices + 1

    pad_list = list()
    for i, seq_len in enumerate(inclusive_break_indices):
        pad_list.append(sequences[i, lower_indices[i] : seq_len])

    packed_seqs = rnn.pack_sequence(pad_list, enforce_sorted=False)

    return packed_seqs


def get_packed_sequence_values(packed_sequence):
    values = (
        packed_sequence.data.unsqueeze(0).unsqueeze(0),
        packed_sequence.batch_sizes.unsqueeze(0),
        packed_sequence.sorted_indices.unsqueeze(0),
        packed_sequence.unsorted_indices.unsqueeze(0),
    )
    return values


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_trajectronpp_pytorch():
    env = load_env()
    hyperparams = load_hyperparams()
    model_dir = "forge/test/models/pytorch/multimodal/trajectron/trajectron/model_dir"

    # Build Pytorch Model
    pt_model = TrajectronWrapper(model_dir=model_dir, hyperparams=hyperparams, env=env, scene_index=0)
    pt_model.eval()

    scene = env.scenes[0]
    inputs_batch = pt_model.get_input_batch(scene=scene)

    (x, x_st_t, first_history_index, neighbors_data_st, neighbors_edge_value, robot_traj_st_t, map), (
        nodes,
        timesteps_o,
    ) = inputs_batch

    packed_x_st_t = pack_input_sequences(x_st_t, lower_indices=first_history_index)
    (
        packed_sequence_data,
        packed_sequence_batch_sizes,
        packed_sequence_sorted_indices,
        packed_sequence_unsorted_indices,
    ) = get_packed_sequence_values(packed_x_st_t)

    assert neighbors_data_st is None
    assert neighbors_edge_value is None
    assert robot_traj_st_t is None
    assert map is None
    # Run CPU Inference
    output = pt_model(
        x,
        x_st_t,
        packed_sequence_data,
        packed_sequence_batch_sizes,
        packed_sequence_sorted_indices,
        packed_sequence_unsorted_indices,
        first_history_index,
    )
    inputs = [
        x,
        x_st_t,
        packed_sequence_data,
        packed_sequence_batch_sizes,
        packed_sequence_sorted_indices,
        packed_sequence_unsorted_indices,
        first_history_index,
    ]
    compiled_model = forge.compile(pt_model, inputs)
    co_out = compiled_model(*inputs)
    fw_out = pt_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out

    assert all([compare_with_golden(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])
