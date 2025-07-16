import torch
import torch.nn as nn
import forge 
from forge.verify.verify import verify
from loguru import logger

def test_s():

    class SDPA(nn.Module):
        def __init__(self):
            super(SDPA, self).__init__()
            self.dropout = 0.0
            self.is_causal = True



        def forward(self, query, key, value):


            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=self.dropout,
                is_causal=self.is_causal,
            )

            return attn_output


    query = torch.load('query_states.pt')
    key = torch.load('key_states.pt')
    value = torch.load('value_states.pt')

    inputs = [query,key,value]


    logger.info("========================================================")

    logger.info("query.shape={}",query.shape)
    logger.info("key.shape={}", key.shape)
    logger.info("value.shape={}",value.shape)

    logger.info("query.dtype={}",query.dtype)
    logger.info("key.dtype={}", key.dtype)
    logger.info("value.dtype={}",value.dtype)

    logger.info("query={}",query)
    logger.info("key={}", key)
    logger.info("value={}",value)

    logger.info("========================================================")

    # Initialize model
    framework_model = SDPA()

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    # Model Verification
    verify(inputs, framework_model, compiled_model)