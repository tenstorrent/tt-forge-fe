# Using the AI Compiler Optimizer with TT-Forge-FE

TT-Forge-FE uses the TT-MLIR AI compiler optimizer. It is responsible for optimizing how operations run on a Tenstorrent chip for maximum performance. Its key purposes are optimizing tensor memory layouts and selecting optimal operation configurations. This page provides resources for using it in code for deploying a trained model with TT-Forge-FE. 

## Prerequisites

To use the AI compiler optimizer with TT-Forge-FE, you need the following: 
* TT-Forge set up, using the [Getting Started with Building TT-Forge-FE From Source](getting_started_build_from_source.md) instructions
* A physical Tenstorrent device 

You can review the [Optimizer page for TT-MLIR](https://docs.tenstorrent.com/tt-mlir/optimizer.html) for further details about how it works with TT-MLIR.

## Configuring the AI Compiler Optimizer for TT-Forge-FE
This section covers how to use the AI Compiler optimizer with TT-Forge-FE. The [**resnet_hf.py**](https://github.com/tenstorrent/tt-forge-fe/blob/odjuricic/simple-perf-test/forge/test/benchmark/benchmark/models/resnet_hf.py) benchmarking sample is used. 

The key part of the code for the optimizer is this: 

```python
    # Turn on MLIR optimizations
    compiler_cfg.mlir_config = (
        MLIRConfig()
        .set_enable_optimizer(True)
        .set_enable_fusing(True)
        .set_enable_fusing_conv2d_with_multiply_pattern(True)
        .set_enable_memory_layout_analysis(False)
    )
```

After you configure your compiler, you can call `compiler_cfg.mlir_config` to set up the optimizer. By default, the optimizer is off. In this example, the following configuration options are used: 
* `.set_enable_optimizer(True)` - Enables the optimizer for use. 
* `.set_enable_fusing(True)` - Enables general operator fusion. This reduces kernel launches, lowers memory reads/writes, and improves cache usage. 
* `.set_enable_fusing_conv2d_with_multiply_pattern(True)` - This specifically looks for `Conv2D` followed by `Multiply` and tries to fuse them into a single kernel. 
* `.set_enable_memory_layout_analysis(False)` - This disables the compiler from optimizing or transforming how tensors are laid out in memory. 

## Running resnet_hf.py