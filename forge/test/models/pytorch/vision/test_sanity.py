import pytest
import torch
import torch.nn as nn
import forge
from loguru import logger
import os

def test_stack_issue1():
    class stack(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_classes=3
            self.num_anchors = 3
            
            
        def forward(self,x):
            
            self.img_size = 608
            num_samples, _, _, grid_size = x.size()

            prediction = x.view(num_samples, self.num_anchors, self.num_classes + 7, grid_size, grid_size)
            prediction = prediction.permute(0, 1, 3, 4, 2).contiguous()
            
            # return prediction
            
            # Get outputs
            pred_x = prediction[..., 0]
            pred_y = prediction[..., 1]
            pred_w = prediction[..., 2]  # Width
            pred_h = prediction[..., 3]  # Height
            pred_im = prediction[..., 4]  # angle imaginary part
            pred_re = prediction[..., 5]  # angle real part
            pred_conf = prediction[..., 6]  # Conf
            pred_cls = prediction[..., 7:]  # Cls pred.
            
            pred_boxes = torch.stack([
                  pred_x,                     
                  pred_y ,                   
                  pred_w,  
                  pred_h,
                  pred_im,                                 
                  pred_re                                  
              ], dim=-1)

            # return pred_boxes
            
            output = torch.cat(
                (
                    pred_boxes[..., :4].view(1, -1, 4) ,
                    pred_boxes[..., 4:6].view(1, -1, 2),
                    pred_conf.view(1, -1, 1),
                    pred_cls.view(1, -1, 3),
                ),
                dim=-1,
            )

            return output
        
    ip = torch.randn(1, 30, 19, 19)
       
    inputs = [ip]
    
    framework_model= stack()
    framework_model.eval()
    os.environ["FORGE_EXTRACT_UNIQUE_OP_CONFIG_AT"] = "ALL"
    os.environ["FORGE_PRINT_UNIQUE_OP_CONFIG"] = "1"
    
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)