# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name
from test.utils import download_model


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, pixel_values, image_sizes):
        return self.model(input_ids, attention_mask, None, None, None, pixel_values, image_sizes)


variants = ["microsoft/Phi-3.5-vision-instruct"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_phi3_5_vision(record_forge_property, variant):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="phi3_5_vision",
        variant=variant,
        task=Task.MULTIMODAL_TEXT_GENERATION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    record_forge_property("tags.model_name", module_name)

    # Load model and processor
    model = download_model(
        AutoModelForCausalLM.from_pretrained,
        variant,
        return_dict=False,
        trust_remote_code=True,
        use_cache=False,
        _attn_implementation="eager",
    )
    model.eval()
    framework_model = Wrapper(model)
    processor = download_model(AutoProcessor.from_pretrained, variant, trust_remote_code=True, num_crops=4)

    # prepare input
    url = "https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-1-2048.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    placeholder = "<|image_1|>\n"
    messages = [{"role": "user", "content": placeholder + "Summarize the slide."}]
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt, image, return_tensors="pt")
    inputs = [inputs["input_ids"], inputs["attention_mask"], inputs["pixel_values"], inputs["image_sizes"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
