# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# BART Demo Script - SQuADv1.1 QA
from test.utils import download_model

import pytest
import torch
from transformers import BartForSequenceClassification, BartTokenizer
from transformers.models.bart.modeling_bart import shift_tokens_right

import forge
from forge.config import CompileDepth, _get_global_compiler_config


class BartWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        out = self.model(input_ids, attention_mask, decoder_input_ids)[0]
        return out


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_pt_bart_classifier(test_device):
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.compile_depth = CompileDepth.SPLIT_GRAPH

    model_name = f"facebook/bart-large-mnli"
    model = download_model(BartForSequenceClassification.from_pretrained, model_name, torchscript=True)
    tokenizer = download_model(BartTokenizer.from_pretrained, model_name, pad_to_max_length=True)
    hypothesis = "Most of Mrinal Sen's work can be found in European collections."
    premise = "Calcutta seems to be the only other production center having any pretensions to artistic creativity at all, but ironically you're actually more likely to see the works of Satyajit Ray or Mrinal Sen shown in Europe or North America than in India itself."

    # generate inputs
    inputs_dict = tokenizer(
        premise,
        hypothesis,
        truncation=True,
        padding="max_length",
        max_length=256,
        truncation_strategy="only_first",
        return_tensors="pt",
    )
    decoder_input_ids = shift_tokens_right(
        inputs_dict["input_ids"], model.config.pad_token_id, model.config.decoder_start_token_id
    )
    inputs = [inputs_dict["input_ids"], inputs_dict["attention_mask"], decoder_input_ids]

    # Compile & feed data
    pt_mod = BartWrapper(model.model)

    compiled_model = forge.compile(pt_mod, sample_inputs=inputs, module_name="pt_bart")
