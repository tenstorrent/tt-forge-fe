"""
Auto-pad support for ONNX operations.
Handles SAME_UPPER, SAME_LOWER, and VALID padding modes.
"""
import torch
import torch.nn.functional as F
from typing import Tuple, Union


class AutoPad:
    """
    Handles automatic padding computation for ONNX auto_pad attribute.
    This is used as a wrapper/preprocessing step before Conv/Pool operations.
    """
    
    @staticmethod
    def compute_padding(
        input_size: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        mode: str = "SAME_UPPER"
    ) -> Tuple[int, int]:
        """
        Compute padding values for a single dimension.
        
        Args:
            input_size: Input size in this dimension
            kernel_size: Kernel size in this dimension
            stride: Stride in this dimension
            dilation: Dilation in this dimension
            mode: "SAME_UPPER", "SAME_LOWER", or "VALID"
            
        Returns:
            Tuple of (pad_before, pad_after)
        """
        if mode == "VALID":
            return (0, 0)
        
        # Calculate effective kernel size with dilation
        effective_kernel = (kernel_size - 1) * dilation + 1
        
        # Calculate output size (ceil division)
        output_size = (input_size + stride - 1) // stride
        
        # Total padding needed
        total_pad = max(0, (output_size - 1) * stride + effective_kernel - input_size)
        
        if mode == "SAME_UPPER":
            # More padding on the right/bottom
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
        else:  # SAME_LOWER
            # More padding on the left/top
            pad_after = total_pad // 2
            pad_before = total_pad - pad_after
        
        return (pad_before, pad_after)
    
    @staticmethod
    def apply_padding(
        x: torch.Tensor,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]] = 1,
        dilation: Union[int, Tuple[int, ...]] = 1,
        mode: str = "SAME_UPPER"
    ) -> torch.Tensor:
        """
        Apply automatic padding to input tensor.
        
        Args:
            x: Input tensor (B, C, H, W) for 2D or (B, C, H) for 1D
            kernel_size: Kernel size (int or tuple)
            stride: Stride (int or tuple)
            dilation: Dilation (int or tuple)
            mode: "SAME_UPPER", "SAME_LOWER", or "VALID"
            
        Returns:
            Padded tensor
        """
        if mode == "VALID":
            return x
        
        # Normalize to tuples
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,)
        if isinstance(stride, int):
            stride = (stride,)
        if isinstance(dilation, int):
            dilation = (dilation,)
        
        # Get spatial dimensions (skip batch and channel)
        spatial_dims = x.shape[2:]
        ndim = len(spatial_dims)
        
        # Compute padding for each spatial dimension
        pads = []
        for i in range(ndim):
            input_size = spatial_dims[i]
            k = kernel_size[i] if i < len(kernel_size) else kernel_size[-1]
            s = stride[i] if i < len(stride) else stride[-1]
            d = dilation[i] if i < len(dilation) else dilation[-1]
            
            pad_before, pad_after = AutoPad.compute_padding(
                input_size, k, s, d, mode
            )
            pads.extend([pad_before, pad_after])
        
        # PyTorch F.pad expects padding in reverse order (last dim first)
        # For 2D: [left, right, top, bottom] -> [left, right, top, bottom]
        # But we need to reverse the order: [pad_H_before, pad_H_after, pad_W_before, pad_W_after]
        pads_reversed = []
        for i in range(ndim - 1, -1, -1):
            pads_reversed.extend([pads[i * 2], pads[i * 2 + 1]])
        
        # Apply padding if needed
        if any(p > 0 for p in pads_reversed):
            x = F.pad(x, pads_reversed, mode='constant', value=0)
        
        return x

