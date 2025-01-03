# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from diffusers import DiffusionPipeline
import torch
import forge
from torchvision import transforms


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
        # Convert the image to PyTorch tensor and detach gradients
        image_tensors = [self.transform(image).detach() for image in images]
        # Stack the tensors into a single tensor
        return torch.stack(image_tensors).detach()


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.xfail(
    reason="RuntimeError: Cannot insert a Tensor that requires grad as a constant. Consider making it a parameter or input, or detaching the gradient"
)
def test_stable_diffusion_generation():
    # Load the pipeline and set it to use the CPU
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32  # Use float32 for CPU
    )

    pipe.to("cpu")  # Move the model to CPU

    # Wrap the pipeline in the wrapper
    model = StableDiffusionXLWrapper(pipe)

    # Tokenize the prompt to a tensor
    tokenizer = pipe.tokenizer
    prompt = "An astronaut riding a green horse"
    input_tensor = tokenizer(prompt, return_tensors="pt").input_ids

    # Generate the image using the wrapper
    images = model(input_tensor)

    # Assert that the image tensor is generated
    assert images is not None, "Image generation failed"
    assert images.shape[0] > 0, "No images were generated"

    # Define the variant (model name) to be used in the module_name
    model_name = "stabilityai/stable-diffusion-xl-base-1.0"

    # Compile the model with tensor input
    compiled_model = forge.compile(
        model,
        sample_inputs=(input_tensor,),  # Pass tensor inputs as tuple
        module_name=f"pt_{model_name.replace('/', '_').replace('.', '_').replace('-', '_')}",
    )
