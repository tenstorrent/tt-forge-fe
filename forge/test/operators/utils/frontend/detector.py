# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Detect xla mode

from loguru import logger

FrontendDetector.XLA_MODE = False
XLA_MODE = FrontendDetector.XLA_MODE
logger.info(f"XLA_MODE: {XLA_MODE}")

__all__ = ["XLA_MODE"]
