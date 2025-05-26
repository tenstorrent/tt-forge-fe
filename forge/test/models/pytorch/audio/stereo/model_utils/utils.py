# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

from .wrapper import Wrapper


def load_model(variant):
    processor = AutoProcessor.from_pretrained(variant)
    model = MusicgenForConditionalGeneration.from_pretrained(variant)
    model = Wrapper(model)
    return model, processor


def load_inputs(model, processor):
    inputs = processor(
        text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
        padding=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]

    pad_token_id = model.model.generation_config.pad_token_id
    decoder_input_ids = (
        torch.ones((inputs.input_ids.shape[0] * model.model.decoder.num_codebooks, 1), dtype=torch.long) * pad_token_id
    )

    return input_ids, attn_mask, decoder_input_ids
