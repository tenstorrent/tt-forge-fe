# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from transformers import ViTConfig, ViTModel, ViTImageProcessor
import forge


def load_model(model_path="google/vit-base-patch16-224", **kwargs):
    """
    Load a Vision Transformer (ViT) model and its feature extractor.

    Args:
        model_path (str): Path or identifier of the pretrained ViT model.
        **kwargs: Additional configuration options.

    Returns:
        framework_model: Loaded ViT model.
        image_processor: Image processor for image preprocessing.
    """
    # Default config values
    config = ViTConfig.from_pretrained(model_path)

    # Override default config values with kwargs
    config.output_attentions = kwargs.get("output_attentions", False)
    config.output_hidden_states = kwargs.get("output_hidden_states", False)
    config.return_dict = kwargs.get("return_dict", True)

    # Load the ViT model
    framework_model = ViTModel.from_pretrained(model_path, config=config)
    framework_model.eval()

    # Load the feature extractor
    image_processor = ViTImageProcessor.from_pretrained(model_path)

    return framework_model, image_processor
