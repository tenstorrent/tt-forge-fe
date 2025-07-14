import torch
import torch.nn as nn
import forge 
from forge.verify.verify import verify
from loguru import logger

def test_s():

    class SDPAWrapper(nn.Module):
        def __init__(self):
            super(SDPAWrapper, self).__init__()
            self.dropout = 0.0
            self.scaling = 0.125
            self.is_causal = False
            


        def forward(self, query, key, value, causal_mask):
            
            
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=causal_mask,
                dropout_p=self.dropout,
                scale=self.scaling,
                is_causal=self.is_causal,
            )
            
            return attn_output
            
    
    # Dummy inputs
    query = torch.randn(1, 14, 39, 64)
    key = torch.randn(1, 14, 39, 64)
    value = torch.randn(1, 14, 39, 64)
    causal_mask = torch.randn(1, 14, 39, 39)
    
    inputs = [query,key,value,causal_mask]
    
    
    logger.info("========================================================")

    logger.info("query.shape={}",query.shape)
    logger.info("key.shape={}", key.shape)
    logger.info("value.shape={}",value.shape)
    logger.info("causal_mask.shape={}",causal_mask.shape)
    
    logger.info("query.dtype={}",query.dtype)
    logger.info("key.dtype={}", key.dtype)
    logger.info("value.dtype={}",value.dtype)
    logger.info("causal_mask.dtype={}",causal_mask.dtype)
    
    logger.info("query={}",query)
    logger.info("key={}", key)
    logger.info("value={}",value)
    logger.info("causal_mask={}",causal_mask)
    
    logger.info("========================================================")

    # Initialize model
    framework_model = SDPAWrapper()

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
