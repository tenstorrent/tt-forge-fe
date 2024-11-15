# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

import forge
from test.mlir.llama.utils.utils import load_model
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from forge.verify.verify import verify


@pytest.mark.parametrize("model_path", ["openlm-research/open_llama_3b", "meta-llama/Llama-3.2-1B"])
@pytest.mark.push
def test_llama_rotary_emb(model_path):
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

    if model_path == "meta-llama/Llama-3.2-1B":
        pytest.skip("Skipping test for Llama-3.2-1B model, waiting for new transformers version.")

    # Load Llama Model
    llama_model, _ = load_model(model_path)

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

    # Compile the model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs=inputs, compiled_model=compiled_model, framework_model=framework_model)
