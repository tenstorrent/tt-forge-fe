# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

import forge
from test.mlir.llama.utils.utils import load_model, load_llama32
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from forge.op.eval.common import compare_with_golden_pcc


@pytest.mark.parametrize("llama_ver", ["llama 3B", "llama 3.2 1B"])
@pytest.mark.xfail(reason="Waiting for the transformers version to be upgraded")
def test_llama_rotary_emb(llama_ver):
    class Llama_Rotary_Embedding(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.rotary_emb = model.model.layers[0].self_attn.rotary_emb
            self.past_key_values_length = 0
            self.seq_length = 12

        def forward(self, query_states, key_states):
            position_ids = torch.arange(
                self.past_key_values_length,
                self.seq_length + self.past_key_values_length,
                dtype=torch.long,
            )
            position_ids = position_ids.unsqueeze(0)
            cos, sin = self.rotary_emb(key_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            return query_states, key_states

    # Load the model
    if llama_ver == "llama 3B":
        llama_model, _ = load_model()
    elif llama_ver == "llama 3.2 1B":
        llama_model, _ = load_llama32()
    framework_model = Llama_Rotary_Embedding(llama_model)
    framework_model.eval()

    # Input samples
    config = llama_model.config
    batch_size = 1
    q_heads = config.num_attention_heads
    kv_heads = config.num_key_value_heads
    query_seq_len = framework_model.seq_length
    kv_seq_len = framework_model.seq_length
    head_dim = config.hidden_size // config.num_attention_heads
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
