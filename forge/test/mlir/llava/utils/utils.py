# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from transformers import AutoProcessor, LlavaForConditionalGeneration

import forge


def load_llava_model(model_path="llava-hf/llava-1.5-7b-hf"):
    """
    Load the LLaVA model and processor.

    Args:
        model_path (str): Path or name of the pre-trained model.

    Returns:
        Tuple: A tuple containing the framework model and processor.
    """
    model = LlavaForConditionalGeneration.from_pretrained(model_path)
    model = model.eval()

    print(model)

    processor = AutoProcessor.from_pretrained(model_path)

    return model, processor
