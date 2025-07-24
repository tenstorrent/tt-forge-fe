import torch
import torch.nn as nn
from typing import Optional, Tuple
import forge
from forge.verify.verify import verify

def test_sdpa_attention():
    class SDPAAttention(nn.Module):
        def __init__(
            self
        ):
            super().__init__()
            self.dropout = 0.0
            self.scaling = 0.125            


        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            causal_mask: Optional[torch.Tensor],
            dropout: float = 0.0,
            scaling: Optional[float] = None,
            is_causal: Optional[bool] = None,
        ) -> Tuple[torch.Tensor, None]:

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=causal_mask,
                dropout_p=dropout,
                scale=scaling,
                is_causal=False,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()

            return attn_output

    attention_mask = torch.load("./attention_mask_qwen.pt")
    
    def generate_tensor(shape, min_val, max_val):
        return (max_val - min_val) * torch.rand(shape) + min_val

    # Shapes
    shape = (1, 16, 6, 64)

    # Generate query, key, value tensors
    query = generate_tensor(shape, -9.0, 13.0)
    key = generate_tensor(shape, -37.0, 42.0)
    value = generate_tensor(shape, -0.15, 0.15)

    is_causal = None
    causal_mask = attention_mask
    if attention_mask is not None and causal_mask.ndim == 4:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]
    if is_causal is None:
        is_causal = query.shape[2] > 1 and causal_mask is None

    inputs = [query, key, value, causal_mask]
    framework_model = SDPAAttention()

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name="sdpa_attention")

    # Model Verification
    verify(inputs, framework_model, compiled_model)
 