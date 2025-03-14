# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from loguru import logger
import random
import requests
import time
import os
import shutil
import urllib

import numpy as np
import torch
import tensorflow as tf


def download_model(download_func, *args, num_retries=3, timeout=180, **kwargs):
    for _ in range(num_retries):
        try:
            return download_func(*args, **kwargs)
        except (
            requests.exceptions.HTTPError,
            urllib.error.HTTPError,
            requests.exceptions.ReadTimeout,
            urllib.error.URLError,
        ):
            logger.trace("HTTP error occurred. Retrying...")
            shutil.rmtree(os.path.expanduser("~") + "/.cache", ignore_errors=True)
            shutil.rmtree(os.path.expanduser("~") + "/.torch/models", ignore_errors=True)
            shutil.rmtree(os.path.expanduser("~") + "/.torchxrayvision/models_data", ignore_errors=True)
            os.mkdir(os.path.expanduser("~") + "/.cache")
        time.sleep(timeout)

    logger.error("Failed to download the model after multiple retries.")
    assert False, "Failed to download the model after multiple retries."


def reset_seeds():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    tf.random.set_seed(0)
