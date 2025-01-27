# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from diffusers import DiffusionPipeline
from torchvision import transforms

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name


class StableDiffusionXLWrapper(torch.nn.Module):
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        # Transformation to convert PIL Image to tensor
        self.transform = transforms.Compose([transforms.ToTensor()])  # Convert PIL Image to PyTorch tensor

    def forward(self, input_tensor):
        # Decode the tensor to text prompt
        tokenizer = self.pipeline.tokenizer
        prompt = tokenizer.decode(input_tensor[0].tolist())
        # Generate images using the pipeline
        images = self.pipeline(prompt=prompt).images
        # return images
        return images


@pytest.mark.nightly
@pytest.mark.skip_model_analysis
@pytest.mark.parametrize("variant", ["stable-diffusion-xl-base-1.0"])
def test_stable_diffusion_generation(record_forge_property, variant):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="stereo",
        variant=variant,
        task=Task.MUSIC_GENERATION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    record_forge_property("model_name", module_name)

    # Load the pipeline and set it to use the CPU
    pipe = DiffusionPipeline.from_pretrained(f"stabilityai/{variant}", torch_dtype=torch.float32)  # Use float32 for CPU
    pipe.to("cpu")  # Move the model to CPU

    # Wrap the pipeline in the wrapper
    framework_model = StableDiffusionXLWrapper(pipe)

    # Tokenize the prompt to a tensor
    tokenizer = pipe.tokenizer
    prompt = "An astronaut riding a green horse"
    input_tensor = tokenizer(prompt, return_tensors="pt").input_ids

    inputs = [input_tensor]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
