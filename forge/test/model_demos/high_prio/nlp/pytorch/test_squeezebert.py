# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
from forge.verify.backend import verify_module
from forge import VerifyConfig
from forge._C.backend_api import BackendType, BackendDevice
from forge.verify.config import TestKind, NebulaGalaxy

import os

import forge
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def test_squeezebert_sequence_classification_pytorch(test_device):
    # Load Bart tokenizer and model from HuggingFace
    tokenizer = download_model(AutoTokenizer.from_pretrained, "squeezebert/squeezebert-mnli")
    model = download_model(AutoModelForSequenceClassification.from_pretrained, "squeezebert/squeezebert-mnli")

    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.default_df_override = forge._C.DataFormat.Float16_b

    # Example from multi-nli validation set
    text = """Hello, my dog is cute"""

    # Data preprocessing
    input_tokens = tokenizer.encode(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    verify_module(
        forge.PyTorchModule("pt_bart", model),
        input_shapes=[(input_tokens.shape,)],
        inputs=[(input_tokens,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            chip_ids=NebulaGalaxy.chip_ids
            if "FORGE_NEB_GALAXY_CI" in os.environ and int(os.environ.get("FORGE_NEB_GALAXY_CI")) == 1
            else [0],
        ),
    )
