
from transformers.models.swinv2.modeling_swinv2 import window_partition
import torch
import torch.nn as nn
import forge
import pytest

@pytest.mark.parametrize("inplace", [True, False])
def test_attn_mask(inplace):

    class Attn_mask_model(nn.Module):
        def __init__(self, inplace=True):
            super().__init__()
            self.inplace = inplace
            self.batch_size = 64
            self.num_attention_heads = 3
            self.dim = 64
            self.shift_size =  4
            self.window_size = 8
            self.mask_shape = 64
            self.height_pad = 64
            self.width_pad=  64


        def get_attn_mask(self, height, width, dtype):

            # calculate attention mask for shifted window multihead self attention
            img_mask = torch.zeros((1, height, width, 1), dtype=dtype)
            
            if self.inplace == True :
                height_slices = (
                    slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None),
                )
                width_slices = (
                    slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None),
                )
                count = 0
                for height_slice in height_slices:
                    for width_slice in width_slices:
                        img_mask[:, height_slice, width_slice, :] = count
                        count += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

            return attn_mask



        def forward(self, attention_scores):

            attention_mask = self.get_attn_mask(self.height_pad, self.width_pad, dtype = torch.float32)
            op = attention_scores.view(
                self.batch_size // self.mask_shape, self.mask_shape, self.num_attention_heads, self.dim, self.dim
            ) + attention_mask.unsqueeze(1).unsqueeze(0)


            return op

    
    inputs = [torch.load('attention_scores.pt')]

    model = Attn_mask_model(inplace=inplace)
    model.eval()

    # Forge compile framework model
    compiled_model = forge.compile(model, sample_inputs=inputs)

def test_inplace_update_alone():

    class inplace_update_alone(nn.Module):
        def __init__(self):
            super().__init__()
            self.shift_size =  4
            self.window_size = 8

        def forward(self, img_mask):


            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1


            return img_mask

    
    inputs = [torch.zeros((1, 64, 64, 1), dtype=torch.float32)]
    model = inplace_update_alone()
    model.eval()

    # Forge compile framework model
    compiled_model = forge.compile(model, sample_inputs=inputs)

    
def test_tensor_creation_and_inplace_update():

    class tensor_creation_and_inplace_update(nn.Module):
        def __init__(self):
            super().__init__()
            self.shift_size =  4
            self.window_size = 8

        def forward(self,x ):

            img_mask = torch.zeros((1, 64, 64, 1), dtype=torch.float32)
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1


            return img_mask + x

    
    inputs = [torch.randn((1, 64, 64, 1), dtype=torch.float32)]
    model = tensor_creation_and_inplace_update()
    model.eval()

    # Forge compile framework model
    compiled_model = forge.compile(model, sample_inputs=inputs)

    
    
