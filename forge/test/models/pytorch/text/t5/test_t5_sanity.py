# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import forge
import torch
import sys
from loguru import logger
from forge import Tensor
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from test.utils import download_model


@pytest.mark.nightly
def test_t5_generation(test_device):
    variant = "google/flan-t5-large"

    config = download_model(T5Config.from_pretrained, variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False

    logger.info("*****************  before  ********************************")
    logger.info("variant={}", variant)
    logger.info("config.num_layers={}", config.num_layers)
    logger.info("config.num_decoder_layers={}", config.num_decoder_layers)
    logger.info("======================================================")

    config_dict["num_decoder_layers"] = 23

    config = T5Config(**config_dict)

    # logger.info("*******************   after  ***********************************")
    # logger.info("variant={}",variant)
    # logger.info("config.num_layers={}",config.num_layers)
    # logger.info("config.num_decoder_layers={}",config.num_decoder_layers)
    # logger.info("======================================================")

    model = download_model(T5ForConditionalGeneration.from_pretrained, variant, config=config)

    # Wrapper to get around attention mask
    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()

            self.model = model.decoder

        def forward(self, decoder_input_ids, encoder_outputs):

            r = self.model(
                decoder_input_ids, None, encoder_outputs, None, None, None, None, None, False, None, None, False
            )

            return r

    decoder_input_ids = torch.randint(0, model.config.vocab_size, (1, 1), dtype=torch.int32)

    encoder_outputs = torch.randn(1, 256, 1024)

    # decoder_input_ids = torch.load('input_ids.pt')

    # encoder_outputs =torch.load('encoder_hidden_states.pt')

    inputs = [decoder_input_ids, encoder_outputs]

    model = Wrapper(model)

    logger.info("model={}", model)

    compiled_model = forge.compile(model, sample_inputs=inputs, module_name="debug_t5_decoder1")
