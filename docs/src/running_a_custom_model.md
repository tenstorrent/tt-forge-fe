# Running a Custom Model with TT-Forge-FE
This walkthrough goes over the example code from the [Getting Started](#getting_started.md) page. It assumes you set up one of the options from there, and focuses on other options for models you can run, and a review of the code sample there.

## Running Demo Models
You can try one of the models in the TT-Forge repo. The following models are currently available:

| Model | Model Type | Description | Demo Code |
|-------|------------|-------------|------------|
| MobileNetV2 | CNN | Lightweight convolutional neural network for efficient image classification | [`cnn/mobile_netv2_demo.py`](cnn/mobile_netv2_demo.py) |
| ResNet-50 | CNN | Deep residual network for image classification | [`cnn/resnet_50_demo.py`](cnn/resnet_50_demo.py) |
| ResNet-50 (ONNX) | CNN | Deep residual network for image classification using ONNX format | [`cnn/resnet_onnx_demo.py`](cnn/resnet_onnx_demo.py) |
| BERT | NLP | Bidirectional Encoder Representations from Transformers for natural language understanding tasks | [`nlp/bert_demo.py`](nlp/bert_demo.py) |

To run one of these models, do the following:

1. Clone the tt-forge repo (alternatively, you can download the script for the model you want to try):

```bash
git clone https://github.com/tenstorrent/tt-forge.git
```

2. In this walkthrough, **resnet_50_demo.py** is used. To run this script, navigate into the **/cnn** folder and run the following command:

```bash
python3 resnet_50_demo.py
```

3. If all goes well, you should see an image of a cat, and terminal output where the model predicts what the image is and presents a score indicating how confident it is in its prediction.

## Creating Your Own Model
When creating your own model, the following general structure is recommended. Note the introduction of the `forge.compile` call:

```py
import torch
from transformers import ResNetForImageClassification

def resnet():
    # Load image, pre-process, etc.
    ...

    # Load model (e.g. from HuggingFace)
    framework_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

    # Compile the model using Forge
    compiled_model = forge.compile(framework_model, input_image)

    # Run compiled model
    logits = compiled_model(input_image)

    ...
    # Post-process output, return results, etc.
```

## Code Example Explanation
This section goes through the code sample provided on the [Getting Started](#getting_started.md) page line by line. This content is for newer users.

```python
import torch
import forge

class Add(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, a, b):
    return a + b

a = torch.rand(size=(2, 32, 32))
b = torch.rand(size=(2, 32, 32))

framework_module = Add()
compiled_model = forge.compile(framework_module, sample_inputs=[a, b])

out = compiled_model(a, b)

print("compiled output:", out)
```

1. ```python
    import torch
    ```

The ```torch``` package is the foundational library of PyTorch. It provides core data structures (tensors) and fundamental operations for numerical computation, particularly for deep learning tasks and leveraging the GPU (Graphics Processing Unit). A tensor is a multi-dimensional array  that can be easily moved to operate on GPUs, which speeds up computations. In PyTorch, tensors integrate with the autograd engine, which automatically computes gradients of operations performed on them. This is key for backpropagation, which is an algorithm in machine learning that is used for training neural networks.

2. ```python
    import forge
    ```

The TT-Forge package bridges frameworks and hardware, acting as a translator and allowing for models from any framework to be converted for use with Tenstorrent devices.

3. ```python
    class Add(torch.nn.Module):
        def __init__(self):
            super().__init__()
    ```

In the next few lines, the code defines a Python class named ```Add``` that inherits from the `torch.nn.Module`. When using PyTorch, all neural network layers and models
