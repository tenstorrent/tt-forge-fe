# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
from transformers.modeling_outputs import (
    MaskedLMOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

# Wrapper to return tensor outputs
class DistilBertWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Case 1: Question Answering
        if isinstance(output, QuestionAnsweringModelOutput):
            return output.start_logits, output.end_logits

        # Case 2: Token Classification, MaskedLM, Sequence Classification
        if isinstance(output, (TokenClassifierOutput, MaskedLMOutput, SequenceClassifierOutput)):
            return output.logits
