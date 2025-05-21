# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import requests
from PIL import Image
from transformers import (
    AutoTokenizer,
    FuyuConfig,
    FuyuForCausalLM,
    FuyuImageProcessor,
    FuyuProcessor,
)

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.pytorch.text.fuyu.model_utils.model import (
    FuyuModelWrapper,
    generate_fuyu_embedding,
)


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "adept/fuyu-8b",
            marks=[pytest.mark.skip(reason="Transient failure - Out of memory due to other tests in CI pipeline")],
        ),
    ],
)
def test_fuyu8b(forge_property_recorder, variant):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH, model="fuyu", variant=variant, task=Task.QA, source=Source.HUGGINGFACE
    )

    config = FuyuConfig.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config_dict["text_config"]["num_hidden_layers"] = 1
    config = FuyuConfig(**config_dict)

    # Load post-processing modules  (run on CPU)
    tokenizer = AutoTokenizer.from_pretrained(variant)
    image_processor = FuyuImageProcessor()
    processor = FuyuProcessor(image_processor=image_processor, tokenizer=tokenizer)

    # Create Forge module from PyTorch model
    fuyu_model = FuyuForCausalLM.from_pretrained(variant, config=config)
    # fuyu_model = FuyuForCausalLM(config=config)
    framework_model = FuyuModelWrapper(fuyu_model)
    framework_model.eval()

    # Prepare inputs
    text_prompt = "Generate a coco-style caption.\n"

    url = "https://huggingface.co/adept-hf-collab/fuyu-8b/resolve/main/bus.png"
    response = requests.get(url)
    with open("bus.png", "wb") as file:
        file.write(response.content)

    image_path = "bus.png"  # https://huggingface.co/adept-hf-collab/fuyu-8b/blob/main/bus.png

    image_pil = Image.open(image_path)
    model_inputs = processor(text=text_prompt, images=[image_pil], device="cpu", return_tensor="pt")
    inputs_embeds = generate_fuyu_embedding(
        fuyu_model, model_inputs["input_ids"], model_inputs["image_patches"][0], model_inputs["image_patches_indices"]
    )
    inputs_embeds = inputs_embeds.clone().detach()

    inputs = [inputs_embeds]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)

    os.remove("bus.png")
