# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from forge.verify.compare import compare_tensor_to_golden
from loguru import logger


def test_check():

    golden = torch.load("golden_k_jan27.pt")
    calculated = torch.load("calculated_k_jan27.pt")

    torch.set_printoptions(linewidth=1000, edgeitems=10, precision=20)

    logger.info(
        "============================== EXISTING PCC CALCULATION ==============================================="
    )

    logger.info("compare_tensor_to_golden={}", compare_tensor_to_golden("existing calculation", golden, calculated))

    logger.info("============ PCC CALCULATION USING TORCH ALL CLOSE with rtol = 1e-5 & atol = 1e-8 ================")

    logger.info("torch.allclose(golden,calculated)={}", torch.allclose(golden, calculated))
