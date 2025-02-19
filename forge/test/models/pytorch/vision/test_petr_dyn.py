import torch
import pytest
from loguru import logger
import forge
import torch.nn as nn

@pytest.mark.parametrize(
    "shape, mask_shape",
    [
        ((300,9),(300,)),
    ],
)
def test_index_issue(shape,mask_shape):
    
    class index_issue(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self,final_box_preds, mask ):
            boxes3d = final_box_preds[mask]
            return boxes3d
 
    input_tensor = torch.rand(shape)
    mask = torch.randint(0, 2, mask_shape, dtype=torch.bool)
    
    logger.info("our inputs")
    logger.info("input_tensor={}",input_tensor)
    logger.info("mask={}",mask)
    logger.info("input_tensor.shape={}",input_tensor.shape)
    logger.info("mask.shape={}",mask.shape)
    
    framework_model = index_issue()
    
    inputs = [input_tensor,mask]
    
     # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
