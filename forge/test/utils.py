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


def fetch_model(model_name, model_path, url, max_retries=3, timeout=30):
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Downloading {model_name} model, attempt {attempt}/{max_retries}...")
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()

            with open(f"{model_path}", "wb") as f:
                f.write(response.content)

            return True

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt}/{max_retries} failed: {str(e)}")

            if attempt < max_retries:
                wait_time = 2**attempt  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {model_name} after {max_retries} attempts")
                return False

    return False


def remove_model(model_path):
    try:
        if os.path.exists(model_path):
            os.remove(model_path)
            return True
    except Exception as e:
        print(f"Error removing model file {model_path}: {str(e)}")
        return False


def reset_seeds():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    tf.random.set_seed(0)
