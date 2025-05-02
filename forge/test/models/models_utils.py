# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import requests
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor
import torch
from tabulate import tabulate
import json

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def preprocess_input_data(image_url):
    input_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    input_tensor = transforms.ToTensor()(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch


def build_optimum_cli_command(variant, tmp_path):
    """
    Constructs the command list for exporting a model using optimum-cli.

    Args:
        variant (str): The model variant to export.
        tmp_path (str): The temporary path where the model will be saved.

    Returns:
        list: A list of command components.
    """
    return [
        "optimum-cli",
        "export",
        "onnx",
        "--model",
        variant,
        tmp_path,
        "--opset",
        "17",
        "--monolith",
        "--framework",
        "pt",
        "--trust-remote-code",
        "--task",
        "text-generation",
        "--library-name",
        "transformers",
        "--batch_size",
        "1",
        "--legacy",
    ]


def get_sample_data(model_name):
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    return [pixel_values]


imagenet_class_index_path = "forge/test/models/files/labels/imagenet_class_index.json"


def load_class_labels(file_path):
    """Load class labels from the local JSON file."""
    with open(file_path, "r") as f:
        class_idx = json.load(f)
    return [class_idx[str(i)][1] for i in range(len(class_idx))]


def print_cls_results(fw_out, compiled_model_out):

    class_labels = load_class_labels(imagenet_class_index_path)
    fw_top1_probabilities, fw_top1_class_indices = torch.topk(fw_out.softmax(dim=1) * 100, k=1)
    compiled_model_top1_probabilities, compiled_model_top1_class_indices = torch.topk(
        compiled_model_out.softmax(dim=1) * 100, k=1
    )

    # Directly get the top 1 predicted class and its probability for both models
    fw_top1_class_idx = fw_top1_class_indices[0, 0].item()
    compiled_model_top1_class_idx = compiled_model_top1_class_indices[0, 0].item()
    fw_top1_class_prob = fw_top1_probabilities[0, 0].item()
    compiled_model_top1_class_prob = compiled_model_top1_probabilities[0, 0].item()

    # Get the class labels for top 1 class
    fw_top1_class_label = class_labels[fw_top1_class_idx]
    compiled_model_top1_class_label = class_labels[compiled_model_top1_class_idx]

    # Prepare the results for displaying
    table = [
        ["Metric", "Framework Model", "Compiled Model"],
        ["Top 1 Predicted Class Label", fw_top1_class_label, compiled_model_top1_class_label],
        ["Top 1 Predicted Class Probability", fw_top1_class_prob, compiled_model_top1_class_prob],
    ]
    print(tabulate(table, headers="firstrow", tablefmt="grid"))


def generate_no_cache(max_new_tokens, model, inputs, seq_len, tokenizer):
    """
    Generates text autoregressively without using a KV cache, iteratively predicting one token at a time.
    The function stops generation if the maximum number of new tokens is reached or an end-of-sequence (EOS) token is encountered.

    Args:
        max_new_tokens (int): The maximum number of new tokens to generate.
        model (torch.nn.Module): The language model used for token generation.
        inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len), representing tokenized text.
        seq_len (int): The current sequence length before generation starts.
        tokenizer: The tokenizer used to decode token IDs into text.

    Returns:
        str: The generated text after decoding the new tokens.
    """
    current_pos = seq_len

    for _ in range(max_new_tokens):
        logits = model(inputs)

        # Get only the logits corresponding to the last valid token
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        else:
            raise TypeError(f"Expected logits to be a list or tuple, but got {type(logits)}")
        next_token_logits = logits[:, current_pos - 1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)
        # Stop if EOS token is encountered
        if next_token_id.item() == tokenizer.eos_token_id:
            break

        inputs[:, current_pos] = next_token_id

        current_pos += 1  # Move to next position

    # Decode valid tokens
    valid_tokens = inputs[:, seq_len:current_pos].view(-1).tolist()
    answer = tokenizer.decode(valid_tokens, skip_special_tokens=True)

    return answer


def pad_inputs(inputs, max_new_tokens=512):
    batch_size, seq_len = inputs.shape
    max_seq_len = seq_len + max_new_tokens
    padded_inputs = torch.zeros((batch_size, max_seq_len), dtype=inputs.dtype, device=inputs.device)
    padded_inputs[:, :seq_len] = inputs
    return padded_inputs, seq_len


def _prepare_4d_causal_attention_mask_with_cache_position(
    self,
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    cache_position: torch.Tensor,
    batch_size: int,
    **kwargs,
):
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0

            # Replace Implace Slice Update
            # causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
            #     padding_mask, min_dtype
            # )

            if causal_mask.shape[-1] > mask_length:
                part_1 = causal_mask[:, :, :, :mask_length]
                part_2 = causal_mask[:, :, :, mask_length:]
                part_1 = part_1.masked_fill(padding_mask, min_dtype)
                causal_mask = torch.cat([part_1, part_2], dim=-1)
            else:
                causal_mask = causal_mask.masked_fill(padding_mask, min_dtype)

    return causal_mask
