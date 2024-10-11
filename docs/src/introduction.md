# Introduction

The **TT-Forge FE** is a graph compiler designed to optimize and transform computational graphs for deep learning models, enhancing their performance and efficiency.

Built on top of the [TT-MLIR](https://docs.tenstorrent.com/tt-mlir/) backend, **TT-Forge FE** is an integral component of the [TT-Forge]() project, which provides a comprehensive suite of tools for optimizing and deploying deep learning models on [Tenstorrent](https://tenstorrent.com/) hardware.

Main project goals are:
- Provide abstraction of many different frontend frameworks (PyTorch, TensorFlow, ONNX, etc.)
- Compile many kinds of model architectures without custom modification and with great performance (e.g. Transformers, CNNs, etc.)
- Abstract all Tenstorrent device architectures (e.g. Wormhole, Blackhole, etc.)
