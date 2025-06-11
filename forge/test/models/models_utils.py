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
from torch import nn

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


def slow_forward_static(
    self,
    input_states,
    cache_params,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
):
    batch_size = input_states.shape[0]
    seq_len = getattr(self, "_static_seq_len", input_states.shape[1])
    dtype = input_states.dtype
    # 1. Gated MLP's linear projection
    projected_states = self.in_proj(input_states).transpose(1, 2)  # [batch, 2 * intermediate_size, seq_len]
    hidden_states, gate = projected_states.chunk(2, dim=1)

    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask.unsqueeze(1)

    # 2. Convolution sequence transformation
    if cache_params is not None:
        ssm_state = cache_params.ssm_states[self.layer_idx].clone()
        ssm_state = ssm_state.to(hidden_states.device)
        # use `cache_position.shape[0]` to check whether we are in prefill
        # stage, it's equivalent to check `cache_position[0] == 0`, which
        # breaks dynamo fullgraph constraints
        if cache_position.shape[0] == self.conv_kernel_size:
            conv_state = nn.functional.pad(hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0))

            cache_params.update_conv_state(self.layer_idx, conv_state, cache_position)
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])  # [batch, intermediate_size, seq_len]
        else:
            conv_state = cache_params.update_conv_state(self.layer_idx, hidden_states, cache_position)
            hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
            if self.use_conv_bias:
                hidden_states += self.conv1d.bias
            hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1)  # [batch, intermediate_size, 1] : decoding
    else:
        ssm_state = torch.zeros(
            (batch_size, self.intermediate_size, self.ssm_state_size), device=hidden_states.device, dtype=dtype
        )
        hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])  # [batch, intermediate_size, seq_len]

    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask.unsqueeze(1)

    # 3. State Space Model sequence transformation
    # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
    ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
    time_step, B, C = torch.split(
        ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
    )
    discrete_time_step = self.dt_proj(time_step)  # [batch, seq_len, intermediate_size]
    discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(
        1, 2
    )  # [batch, intermediate_size, seq_len]

    # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
    A = -torch.exp(self.A_log.float())  # [intermediate_size, ssm_state_size]
    discrete_A = torch.exp(
        A[None, :, None, :] * discrete_time_step[:, :, :, None]
    )  # [batch, intermediate_size, seq_len, ssm_state_size]
    discrete_B = (
        discrete_time_step[:, :, :, None] * B[:, None, :, :].float()
    )  # [batch, intermediate_size, seq_len, ssm_state_size]
    deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

    # 3.c perform the recurrence y ← SSM(A, B, C)(x)
    if self.use_mambapy and self.training and cache_params is None:
        hs = pscan(
            discrete_A.transpose(1, 2), deltaB_u.transpose(1, 2)
        )  # [batch, seq_len, intermediate_size, ssm_state_size]

        scan_output = (hs @ C.unsqueeze(-1)).squeeze(3).transpose(1, 2)  # [batch, intermediate_size, seq_len]
        scan_output = scan_output + hidden_states * self.D[None, :, None]
        scan_output = scan_output * self.act(gate)
    else:
        scan_outputs = []
        for i in range(seq_len):
            ssm_state = (
                discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
            )  # [batch, intermediade_size, ssm_state]
            scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))  # [batch, intermediade_size, 1]
            scan_outputs.append(scan_output[:, :, 0])
        scan_output = torch.stack(scan_outputs, dim=-1)  # [batch, seq_len, intermediade_size]
        scan_output = scan_output + (hidden_states * self.D[None, :, None])
        scan_output = scan_output * self.act(gate)

        if cache_params is not None:
            cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

    # 4. Final linear projection
    contextualized_states = self.out_proj(scan_output.transpose(1, 2))  # [batch, seq_len, hidden_size]
    return contextualized_states
