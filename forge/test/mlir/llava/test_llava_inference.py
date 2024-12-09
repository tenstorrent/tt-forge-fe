# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import forge
from test.mlir.llava.utils.utils import load_llava_model


@pytest.mark.nightly
@pytest.mark.parametrize("model_path", ["liuhaotian/LLaVA-7b-v0", "liuhaotian/LLaVA-13b-v0"])
def test_llava_inference(model_path):
    if model_path == "liuhaotian/LLaVA-13b-v0":
        pytest.skip("Skipping test for LLaVA-13b-v0 model, waiting for updated support in transformers.")

    # Load LLaVA Model and Processor
    framework_model, processor = load_llava_model(model_path)

    # Prepare Inputs
    image_path = "musashi.jpg"  # Replace with a valid image path
    question = "What does this image depict?"
    inputs = processor(image=image_path, text=question, return_tensors="pt")

    # Sanity Run
    generated_ids = framework_model.generate(**inputs, max_new_tokens=32)
    response = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)

    # Compile the Model
    compiled_model = forge.compile(framework_model, sample_inputs=(inputs["input_ids"], inputs["pixel_values"]))
