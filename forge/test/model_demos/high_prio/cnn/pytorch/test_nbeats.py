# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import forge
import sys

sys.path.append("forge/test/model_demos/models")

from nbeats.scripts import (
    get_electricity_dataset_input,
    NBeatsWithGenericBasis,
    NBeatsWithTrendBasis,
    NBeatsWithSeasonalityBasis,
)


def test_nbeats_with_seasonality_basis(test_device):
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.FINISH_COMPILE

    x, x_mask = get_electricity_dataset_input()

    pytorch_model = NBeatsWithSeasonalityBasis(
        input_size=72,
        output_size=24,
        num_of_harmonics=1,
        stacks=30,
        layers=4,
        layer_size=2048,
    )
    pytorch_model.eval()
    compiled_model = forge.compile(pytorch_model, sample_inputs=[x, x_mask], module_name="nbeats_seasonality")


def test_nbeats_with_generic_basis(test_device):
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.FINISH_COMPILE

    x, x_mask = get_electricity_dataset_input()

    pytorch_model = NBeatsWithGenericBasis(input_size=72, output_size=24, stacks=30, layers=4, layer_size=512)
    pytorch_model.eval()

    compiled_model = forge.compile(pytorch_model, sample_inputs=[x, x_mask], module_name="nbeats_generic")


def test_nbeats_with_trend_basis(test_device):
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.FINISH_COMPILE

    x, x_mask = get_electricity_dataset_input()

    pytorch_model = NBeatsWithTrendBasis(
        input_size=72,
        output_size=24,
        degree_of_polynomial=3,
        stacks=30,
        layers=4,
        layer_size=256,
    )
    pytorch_model.eval()

    compiled_model = forge.compile(pytorch_model, sample_inputs=[x, x_mask], module_name="nbeats_trend")
