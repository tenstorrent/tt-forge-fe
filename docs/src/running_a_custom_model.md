## Run First Example Case

To confirm that our environment is properly setup, let's run one sanity test for element-wise add operation:
```bash
pytest forge/test/mlir/operators/eltwise_binary/test_eltwise_binary.py::test_add
```

In a few seconds, you should get confirmation if this test passed successfully. Once that's done, we can run one of our model tests as well:
```bash
pytest forge/test/mlir/llama/tests/test_llama_prefil.py::test_llama_prefil_on_device_decode_on_cpu
```

## Running Models

You can try one of the models in the tt-forge repo. For a list of models that work with tt-forge-fe, navigate to the [Demos](https://github.com/tenstorrent/tt-forge/tree/main/demos/tt-forge-fe) folder in the tt-forge repo. Follow the [Getting Started](https://github.com/tenstorrent/tt-forge/blob/main/docs/src/getting-started.md) instructions there.


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
