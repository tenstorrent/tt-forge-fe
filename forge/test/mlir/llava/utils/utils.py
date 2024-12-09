# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from transformers import CLIPProcessor, AutoModelForCausalLM, AutoTokenizer

import forge


def load_llava_model(model_path="liuhaotian/LLaVA-7b-v0", **kwargs):
    """
    Load the LLaVA model and processor.

    Args:
        model_path (str): Path or name of the pre-trained model.
        **kwargs: Additional arguments for customization.

    Returns:
        Tuple: A tuple containing the framework model and processor.
    """
    # Load Text Decoder
    framework_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
    )
    framework_model.eval()

    # Load Image Encoder and Processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    return framework_model, processor
