# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import time

from test.mlir.nerf.utils import NeRF
import torch
from loguru import logger

import forge
from forge.tensor import to_forge_tensors
from forge.verify.compare import compare_with_golden
from test.mlir.nerf.spherical_harmonics import eval_sh
from test.mlir.utils import *


@pytest.mark.push
def test_nerf_training():
    dtype = torch.float32

    # Set training hyperparameters
    num_epochs = 3
    num_batches = 10
    batch_size = 4096
    learning_rate = 1e-3
    deg = 2

    # Define models
    nerf_coarse = NeRF(D=4, W=128, in_channels_xyz=63, in_channels_dir=32, deg=deg)
    nerf_fine = NeRF(D=4, W=192, in_channels_xyz=63, in_channels_dir=32, deg=deg)

    golden_nerf_coarse = NeRF(D=4, W=128, in_channels_xyz=63, in_channels_dir=32, deg=deg)
    golden_nerf_fine = NeRF(D=4, W=192, in_channels_xyz=63, in_channels_dir=32, deg=deg)

    copy_params(nerf_coarse, golden_nerf_coarse)
    copy_params(nerf_fine, golden_nerf_fine)

    loss_fn = torch.nn.MSELoss()

    # Define optimizer
    framework_optimizer_coarse = torch.optim.SGD(nerf_coarse.parameters(), lr=learning_rate)
    framework_optimizer_fine = torch.optim.SGD(nerf_fine.parameters(), lr=learning_rate)
    golden_optimizer_coarse = torch.optim.SGD(golden_nerf_coarse.parameters(), lr=learning_rate)
    golden_optimizer_fine = torch.optim.SGD(golden_nerf_fine.parameters(), lr=learning_rate)

    tt_nerf_coarse = forge.compile(
        nerf_coarse,
        sample_inputs=[torch.rand(batch_size, 63, dtype=dtype, requires_grad=True)],
        optimizer=framework_optimizer_coarse,
        training=True,
    )

    tt_nerf_fine = forge.compile(
        nerf_fine,
        sample_inputs=[torch.rand(batch_size, 63, dtype=dtype, requires_grad=True)],
        optimizer=framework_optimizer_fine,
        training=True,
    )

    logger.info("Starting NeRF training loop... (logger will be disabled)")
    logger.disable("")
    for epoch_idx in range(num_epochs):
        for batch_idx in range(num_batches):
            # zero the parameter gradients
            framework_optimizer_coarse.zero_grad()
            framework_optimizer_fine.zero_grad()
            golden_optimizer_coarse.zero_grad()
            golden_optimizer_fine.zero_grad()

            # Generate random input data
            input_xyz = torch.rand(batch_size, 63, dtype=dtype, requires_grad=True)
            input_dirs = torch.rand(batch_size, 3, dtype=dtype, requires_grad=True)
            target_data_sigma = torch.rand(batch_size, 1, dtype=dtype, requires_grad=True)
            target_data_sh = torch.rand(batch_size, 3, dtype=dtype, requires_grad=True)

            # Forward pass on TT
            output_coarse_sigma, output_coarse_sh = tt_nerf_coarse(input_xyz)
            output_coarse_sh = output_coarse_sh[:, :27].reshape(-1, 3, (deg + 1) ** 2)
            output_coarse_rgb = eval_sh(deg, output_coarse_sh, input_dirs)

            output_fine_sigma, output_fine_sh = tt_nerf_fine(input_xyz)
            output_fine_sh = output_fine_sh[:, :27].reshape(-1, 3, (deg + 1) ** 2)
            output_fine_rgb = eval_sh(deg, output_fine_sh, input_dirs)

            # Forward pass on PyTorch
            golden_output_coarse_sigma, output_coarse_sh_pt = golden_nerf_coarse(input_xyz)
            golden_output_coarse_sh_pt = output_coarse_sh_pt[:, :27].reshape(-1, 3, (deg + 1) ** 2)
            golden_output_coarse_rgb_pt = eval_sh(deg, golden_output_coarse_sh_pt, input_dirs)

            golden_output_fine_sigma, golden_output_fine_sh = golden_nerf_fine(input_xyz)
            golden_output_fine_sh = golden_output_fine_sh[:, :27].reshape(-1, 3, (deg + 1) ** 2)
            golden_output_fine_rgb = eval_sh(deg, golden_output_fine_sh, input_dirs)

            # Compute loss for TT
            loss_coarse_sigma = loss_fn(output_coarse_sigma, target_data_sigma)
            loss_coarse_sh = loss_fn(output_coarse_rgb, target_data_sh)
            loss_coarse = loss_coarse_sigma + loss_coarse_sh

            loss_fine_sigma = loss_fn(output_fine_sigma, target_data_sigma)
            loss_fine_sh = loss_fn(output_fine_rgb, target_data_sh)
            loss_fine = loss_fine_sigma + loss_fine_sh

            # Compute loss for PyTorch
            golden_loss_coarse_sigma = loss_fn(golden_output_coarse_sigma, target_data_sigma)
            golden_loss_coarse_sh = loss_fn(golden_output_coarse_rgb_pt, target_data_sh)
            golden_loss_coarse = golden_loss_coarse_sigma + golden_loss_coarse_sh

            golden_loss_fine_sigma = loss_fn(golden_output_fine_sigma, target_data_sigma)
            golden_loss_fine_sh = loss_fn(golden_output_fine_rgb, target_data_sh)
            golden_loss_fine = golden_loss_fine_sigma + golden_loss_fine_sh

            # Compare TT and PyTorch losses
            assert compare_with_golden(
                loss_coarse, golden_loss_coarse, rtol=0.05, atol=0.05
            ), f"Loss coarse mismatch at epoch {epoch_idx}, batch {batch_idx}"
            assert compare_with_golden(
                loss_fine, golden_loss_fine, rtol=0.05, atol=0.05
            ), f"Loss fine mismatch at epoch {epoch_idx}, batch {batch_idx}"

            # Backward pass
            loss_coarse.backward()
            loss_fine.backward()

            golden_loss_coarse.backward()
            golden_loss_fine.backward()

            # Update weights
            framework_optimizer_coarse.step()
            framework_optimizer_fine.step()

            golden_optimizer_coarse.step()
            golden_optimizer_fine.step()

    logger.enable("")
    logger.info("NeRF training loop completed.")
