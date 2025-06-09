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
from typing import Optional, Tuple
from transformers import Cache
from transformers.models.dpr.modeling_dpr import DPRContextEncoderOutput, DPRQuestionEncoderOutput
from transformers.modeling_outputs import BaseModelOutputWithPooling
from typing import Optional, Tuple, Union
from torch import Tensor

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
        elif isinstance(logits, torch.Tensor):
            logits = logits
        else:
            raise TypeError(f"Expected logits to be a list or tuple or torch.Tensor, but got {type(logits)}")
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


def Gemma2DecoderLayer_patched_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    if self.is_sliding and attention_mask is not None:  # efficient SDPA and no padding
        # Flash-attn is a 2D tensor
        if self.config._attn_implementation == "flash_attention_2":
            if past_key_value is not None:  # when decoding
                attention_mask = attention_mask[:, -self.sliding_window :]
        else:
            # min_dtype = torch.finfo(hidden_states.dtype).min

            # [Monkey patch] Cast scalar to tensor with hidden_states dtype to fix ONNX Where op type mismatch
            min_dtype = torch.tensor(torch.finfo(hidden_states.dtype).min, dtype=hidden_states.dtype)

            sliding_window_mask = torch.tril(
                torch.ones_like(attention_mask, dtype=torch.bool), diagonal=-self.sliding_window
            )
            attention_mask = torch.where(sliding_window_mask, min_dtype, attention_mask)
            if attention_mask.shape[-1] <= 1:  # when decoding
                attention_mask = attention_mask[:, :, :, -self.sliding_window :]

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
    )
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.pre_feedforward_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = self.post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def dpr_encoder_forward_with_position_ids(
    self,
    input_ids: Tensor,
    attention_mask: Optional[Tensor] = None,
    token_type_ids: Optional[Tensor] = None,
    position_ids: Optional[Tensor] = None,
    inputs_embeds: Optional[Tensor] = None,
    output_attentions: bool = False,
    output_hidden_states: bool = False,
    return_dict: bool = False,
) -> Union[BaseModelOutputWithPooling, Tuple[Tensor, ...]]:
    outputs = self.bert_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    sequence_output = outputs[0]
    pooled_output = sequence_output[:, 0, :]

    if self.projection_dim > 0:
        pooled_output = self.encode_proj(pooled_output)

    if not return_dict:
        return (sequence_output, pooled_output) + outputs[2:]

    return BaseModelOutputWithPooling(
        last_hidden_state=sequence_output,
        pooler_output=pooled_output,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def dpr_context_encoder_forward_with_position_ids(
    self,
    input_ids: Optional[Tensor] = None,
    attention_mask: Optional[Tensor] = None,
    token_type_ids: Optional[Tensor] = None,
    position_ids: Optional[Tensor] = None,
    inputs_embeds: Optional[Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[DPRContextEncoderOutput, Tuple[Tensor, ...]]:

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if attention_mask is None:
        attention_mask = (
            torch.ones(input_shape, device=device) if input_ids is None else (input_ids != self.config.pad_token_id)
        )
    if token_type_ids is None:
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    outputs = self.ctx_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    if not return_dict:
        return outputs[1:]
    return DPRContextEncoderOutput(
        pooler_output=outputs.pooler_output, hidden_states=outputs.hidden_states, attentions=outputs.attentions
    )


def dpr_question_encoder_forward_with_position_ids(
    self,
    input_ids: Optional[Tensor] = None,
    attention_mask: Optional[Tensor] = None,
    token_type_ids: Optional[Tensor] = None,
    position_ids: Optional[Tensor] = None,
    inputs_embeds: Optional[Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[DPRQuestionEncoderOutput, Tuple[Tensor, ...]]:

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        input_shape = input_ids.size()
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if attention_mask is None:
        attention_mask = (
            torch.ones(input_shape, device=device) if input_ids is None else (input_ids != self.config.pad_token_id)
        )
    if token_type_ids is None:
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    outputs = self.question_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    if not return_dict:
        return outputs[1:]
    return DPRQuestionEncoderOutput(
        pooler_output=outputs.pooler_output, hidden_states=outputs.hidden_states, attentions=outputs.attentions
    )
