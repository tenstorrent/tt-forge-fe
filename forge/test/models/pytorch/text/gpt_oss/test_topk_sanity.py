# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch.nn.functional as F

import forge
from forge.verify.verify import DeprecatedVerifyConfig, verify


def test_gpt_oss_topk_sanity():
    class GptOssTopKRouter(nn.Module):
        def __init__(self):
            super().__init__()
            self.top_k = 4
            self.num_experts = 32
            self.hidden_dim = 2880
            self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
            self.bias = nn.Parameter(torch.empty(self.num_experts))

        def forward(self, hidden_states):
            hidden_states = hidden_states.reshape(-1, self.hidden_dim)
            router_logits = F.linear(hidden_states, self.weight, self.bias)  # (seq_len, num_experts)
            router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
            router_indices_float = router_indices.float()
            output = torch.matmul(router_top_value.transpose(-2, -1), router_indices_float)
            return output

    # hidden_states = torch.randn(1, 10, 2880)
    hidden_states = torch.load("/proj_sw/user_dev/mramanathan/bgdlab14_aug11_forge/tt-forge-fe/hidden_states_topk.pt")
    model = GptOssTopKRouter()

    # Forge compile framework model
    compiled_model = forge.compile(
        model,
        sample_inputs=[hidden_states],
        module_name="topk_sanity",
        verify_cfg=DeprecatedVerifyConfig(verify_forge_codegen_vs_framework=True),
    )

    # Model Verification
    verify([hidden_states], model, compiled_model)
