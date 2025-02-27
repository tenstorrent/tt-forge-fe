# Getting Started

## Setup
You choose between two ways to setup our project:
- Install pre-built wheel
- Building from source

### Install using Wheel

Download the latest `tt-forge-fe` release, which includes both TVM and Forge wheels:
- https://github.com/tenstorrent/tt-forge/releases

Install both TVM and Forge wheels using following command:
```py
pip install *.whl
```

> Note: Make sure to run the command from the directory where wheels are downloaded.

#### Run Wheel test

To confirm that your wheel is installed properly, you can run this command.

```bash
curl -o forge/test/mlir/test_ops_tf.py https://raw.githubusercontent.com/tenstorrent/tt-forge-fe/main/forge/test/test_api.py
pytest -svv forge/test/
```

### Build from Source

To build Forge-FE from source, you need to clone the project from our GitHub page:
```bash
git clone https://github.com/tenstorrent/tt-forge-fe.git
```

Afterwards, you can follow our [build instructions](https://docs.tenstorrent.com/tt-forge-fe/build.html) which outline prerequisites, as well as how to build dependencies and our project.

#### Run Source build test

To confirm that our environment is properly setup, let's run one sanity test for element-wise add operation:
```bash
pytest forge/test/mlir/operators/eltwise_binary/test_eltwise_binary.py::test_add
```

In a few seconds, you should get confirmation if this test passed successfully. Once that's done, we can run one of our model tests as well:
```bash
pytest forge/test/mlir/llama/tests/test_llama_prefil.py::test_llama_prefil_on_device_decode_on_cpu
```

## Where to Go Next

Now that you have set up Forge-FE, you can try to compile and run your own models!

For a quick start, here is an example of how to run your own model. Note the introduction of the `forge.compile` call:

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
