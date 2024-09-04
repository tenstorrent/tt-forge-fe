# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import forge

from forge._C.backend_api import BackendDevice
from ..common import benchmark_model, generate_test_device
from test.model_demos.models.t5 import generate_t5_past_cache_enc_dec
from forge.config import _get_global_compiler_config


@benchmark_model(configs=["base", "large"])
def t5(training: bool, config: str, microbatch: int, devtype: str, arch: str, data_type: str, math_fidelity: str):

    compiler_cfg = _get_global_compiler_config()

    if compiler_cfg.balancer_policy == "default":
        compiler_cfg.balancer_policy = "Ribbon"
        os.environ["FORGE_RIBBON2"] = "1"

    # These are about to be enabled by default.
    #
    os.environ["FORGE_RIBBON2_CALCULATE_TARGET_CYCLES"] = "1"
    os.environ["FORGE_EXP_APPROX"] = "1"

    if data_type == "Bfp8_b":
        forge.config.configure_mixed_precision(op_type="add", output_df=forge.DataFormat.Float16_b)
        forge.config.configure_mixed_precision(op_type="subtract", output_df=forge.DataFormat.Float16_b)
        forge.config.configure_mixed_precision(op_type="reciprocal", output_df=forge.DataFormat.Float16_b)

    available_devices = forge.detect_available_devices()
    # Determine model variant
    if config == "base":
        variant = "t5-base"

    elif config == "large":
        variant = "t5-large"
    else:
        raise RuntimeError("Unknown config")

    # Load model
    modules, inputs, other = generate_t5_past_cache_enc_dec(
        generate_test_device(devtype, arch),
        variant,
    )

    targets = tuple()

    return modules, inputs, targets, other


@benchmark_model(configs=["base", "large"])
def flan_t5(training: bool, config: str, microbatch: int, devtype: str, arch: str, data_type: str, math_fidelity: str):

    compiler_cfg = _get_global_compiler_config()

    if compiler_cfg.balancer_policy == "default":
        compiler_cfg.balancer_policy = "Ribbon"
        os.environ["FORGE_RIBBON2"] = "1"

    # These are about to be enabled by default.
    #
    os.environ["FORGE_RIBBON2_CALCULATE_TARGET_CYCLES"] = "1"
    os.environ["FORGE_EXP_APPROX"] = "1"

    # Determine model variant
    if config == "base":
        variant = "google/flan-t5-base"
    elif config == "large":
        variant = "google/flan-t5-large"
    else:
        raise RuntimeError("Unknown config")
    
    if data_type == "Bfp8_b":
        forge.config.configure_mixed_precision(op_type="add", output_df=forge.DataFormat.Float16_b)
        forge.config.configure_mixed_precision(op_type="subtract", output_df=forge.DataFormat.Float16_b)
        forge.config.configure_mixed_precision(op_type="reciprocal", output_df=forge.DataFormat.Float16_b)

    # Load model
    modules, inputs, other = generate_t5_past_cache_enc_dec(
        generate_test_device(devtype, arch),
        variant,
    )
 
    targets = tuple()

    return modules, inputs, targets, other
