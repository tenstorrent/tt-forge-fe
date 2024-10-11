# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

import forge
from test.mlir.llama.utils.utils import load_model
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from forge.op.eval.common import compare_with_golden_pcc


@pytest.mark.xfail()
def test_llama_rotary_emb():
    class Llama_Rotary_Embedding(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.rotary_emb = model.model.layers[0].self_attn.rotary_emb
            self.past_key_values_length = 0
            self.seq_length = 12

        def forward(self, query_states, key_states):
            kv_seq_len = key_states.shape[-2]
            cos, sin = self.rotary_emb(key_states, seq_len=kv_seq_len)
            position_ids = torch.arange(
                self.past_key_values_length,
                self.seq_length + self.past_key_values_length,
                dtype=torch.long,
            )
            position_ids = position_ids.unsqueeze(0)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
            return query_states, key_states

    # Load the model
    llama_model = load_model()
    framework_model = Llama_Rotary_Embedding(llama_model)
    framework_model.eval()

    # Input samples
    batch_size, q_heads, kv_heads, query_seq_len, kv_seq_len, head_dim = 1, 32, 32, 12, 12, 100
    inputs = [
        torch.rand((batch_size, q_heads, query_seq_len, head_dim)),  # Query states
        torch.rand((batch_size, kv_heads, kv_seq_len, head_dim)),  # Key states
    ]

    # Sanity run
    fw_out = framework_model(*inputs)

    # Compile the model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    # Run on TT device
    tt_out = compiled_model(*inputs)
    tt_out = [out.to("cpu") for out in tt_out]

    # Validate results
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, tt_out)])
