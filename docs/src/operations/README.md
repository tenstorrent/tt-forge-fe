# Forge Operations Reference

Welcome to the Forge Operations Reference. This page provides a comprehensive guide to all supported operations in the Forge framework.

## Overview

Forge operations are organized into logical categories based on their functionality. Each operation is documented with detailed information including function signatures, parameters, examples, and usage notes.

## Quick Navigation

 - [Elementwise Operations](#elementwise-operations) - Mathematical operations applied element-wise
- [Convolution Operations](#convolution-operations) - Convolution and related transformations
- [Pooling Operations](#pooling-operations) - Pooling and downsampling operations
- [Normalization Operations](#normalization-operations) - Batch and layer normalization
- [Tensor Manipulation](#tensor-manipulation) - Reshaping, slicing, and tensor operations
- [Reduction Operations](#reduction-operations) - Aggregation and reduction operations
- [Linear Operations](#linear-operations) - Matrix multiplication and linear transformations
- [Activation Functions](#activation-functions) - Non-linear activation functions
- [Memory Operations](#memory-operations) - Cache and memory management operations
- [Other Operations](#other-operations) - Miscellaneous operations

---

## Elementwise Operations

Elementwise operations apply a function to each element of the input tensor(s) independently.

| Operation | Description | Link |
|-----------|-------------|------|
| **Abs** | Elementwise absolute value operation | [forge.op.Abs](./operations/forge_op_abs.md) |
| **Add** | Elementwise addition of two tensors | [forge.op.Add](./operations/forge_op_add.md) |
| **Atan** | Elementwise arctangent (atan) | [forge.op.Atan](./operations/forge_op_atan.md) |
| **Subtract** | Elementwise subtraction of two tensors | [forge.op.Subtract](./operations/forge_op_subtract.md) |

## Convolution Operations
Convolution operations perform spatial filtering and feature extraction.

| Operation | Description | Link |
|-----------|-------------|------|
| **Conv2d** | 2D convolution transformation with optional bias | [forge.op.Conv2d](./operations/forge_op_conv2d.md) |
| **Conv2dTranspose** | 2D transposed convolution (deconvolution) | [forge.op.Conv2dTranspose](./operations/forge_op_conv2dtranspose.md) |

## Pooling Operations

Pooling operations reduce spatial dimensions and provide translation invariance.

| Operation | Description | Link |
|-----------|-------------|------|
| **AvgPool1d** | 1D average pooling transformation | [forge.op.AvgPool1d](./operations/forge_op_avgpool1d.md) |
| **AvgPool2d** | 2D average pooling transformation | [forge.op.AvgPool2d](./operations/forge_op_avgpool2d.md) |

## Normalization Operations

Normalization operations stabilize training and improve convergence.

| Operation | Description | Link |
|-----------|-------------|------|
| **Batchnorm** | Batch normalization | [forge.op.Batchnorm](./operations/forge_op_batchnorm.md) |
| **Layernorm** | Layer normalization | [forge.op.Layernorm](./operations/forge_op_layernorm.md) |

## Tensor Manipulation

Operations for reshaping, slicing, and manipulating tensor structure.

| Operation | Description | Link |
|-----------|-------------|------|
| **Reshape** | Reshape tensor to new dimensions | [forge.op.Reshape](./operations/forge_op_reshape.md) |
| **Select** | Select operation | [forge.op.Select](./operations/forge_op_select.md) |
| **Squeeze** | Remove dimensions of size 1 | [forge.op.Squeeze](./operations/forge_op_squeeze.md) |
| **Stack** | Stack tensors along new axis | [forge.op.Stack](./operations/forge_op_stack.md) |
| **Transpose** | Transpose X and Y dimensions | [forge.op.Transpose](./operations/forge_op_transpose.md) |
| **Unsqueeze** | Add dimension of size 1 | [forge.op.Unsqueeze](./operations/forge_op_unsqueeze.md) |
| **AdvIndex** | Advanced indexing operation | [forge.op.AdvIndex](./operations/forge_op_advindex.md) |

## Reduction Operations

Operations that reduce tensor dimensions by aggregation.

| Operation | Description | Link |
|-----------|-------------|------|
| **ReduceAvg** | Reduce by averaging along dimension | [forge.op.ReduceAvg](./operations/forge_op_reduceavg.md) |
| **ReduceMax** | Reduce by taking maximum along dimension | [forge.op.ReduceMax](./operations/forge_op_reducemax.md) |
| **ReduceSum** | Reduce by summing along dimension | [forge.op.ReduceSum](./operations/forge_op_reducesum.md) |

## Linear Operations

Matrix multiplication and linear transformations.

| Operation | Description | Link |
|-----------|-------------|------|
| **Matmul** | Matrix multiplication with optional bias | [forge.op.Matmul](./operations/forge_op_matmul.md) |

## Activation Functions

Non-linear activation functions for neural networks.

| Operation | Description | Link |
|-----------|-------------|------|
| **Gelu** | Gaussian Error Linear Unit (GELU) | [forge.op.Gelu](./operations/forge_op_gelu.md) |
| **Relu** | Rectified Linear Unit (ReLU) | [forge.op.Relu](./operations/forge_op_relu.md) |
| **Sigmoid** | Sigmoid activation function | [forge.op.Sigmoid](./operations/forge_op_sigmoid.md) |
| **Tanh** | Hyperbolic tangent activation | [forge.op.Tanh](./operations/forge_op_tanh.md) |

## Memory Operations

Operations for cache and memory management.

| Operation | Description | Link |
|-----------|-------------|------|
| **FillCache** | Write input into cache tensor at specified index | [forge.op.FillCache](./operations/forge_op_fillcache.md) |
| **UpdateCache** | Write single token slice into cache tensor | [forge.op.UpdateCache](./operations/forge_op_updatecache.md) |

## Other Operations

Miscellaneous operations.

| Operation | Description | Link |
|-----------|-------------|------|
| **Constant** | User-defined constant operation | [forge.op.Constant](./operations/forge_op_constant.md) |
| **Embedding** | Embedding lookup operation | [forge.op.Embedding](./operations/forge_op_embedding.md) |
| **Where** | Conditional element selection | [forge.op.Where](./operations/forge_op_where.md) |

---

## Documentation Structure

Each operation documentation page includes:

- **Overview**: Brief description of what the operation does
- **Function Signature**: Python API signature with type hints
- **Parameters**: Detailed parameter descriptions with types and defaults
- **Returns**: Return value description
- **Mathematical Definition**: Mathematical formula (where applicable)
- **Examples**: Code examples demonstrating usage
- **Notes**: Important implementation details and warnings
- **Related Operations**: Links to related operations

---

*This documentation is automatically generated from operation definitions. For the most up-to-date information, refer to the source code.*
