# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_no_cache(max_new_tokens, model, inputs, tokenizer):
    """
    Generates text tokens autoregressively up to a maximum length by iteratively predicting the next token
    using the model and appending it to the sequence until the limit is reached or an EOS token is encountered.

    Args:
        max_new_tokens (int): Maximum number of tokens to generate.
        model (torch.nn.Module): The language model used for generation.
        inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len).
        tokenizer: Tokenizer for decoding token IDs into text.

    Returns:
        str: The generated text.
    """
    batch_size, seq_len = inputs.shape
    max_seq_len = seq_len + max_new_tokens  # Fixed total sequence length

    padded_inputs = torch.zeros((batch_size, max_seq_len), dtype=inputs.dtype, device=inputs.device)
    padded_inputs[:, :seq_len] = inputs

    current_pos = seq_len

    for _ in range(max_new_tokens):
        logits = model(padded_inputs)

        # Get only the logits corresponding to the last valid token
        if isinstance(logits, list):
            logits = logits[0]
        next_token_logits = logits[:, current_pos - 1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)
        # Stop if EOS token is encountered
        if next_token_id.item() == tokenizer.eos_token_id:
            break

        padded_inputs[:, current_pos] = next_token_id

        current_pos += 1  # Move to next position

    # Decode valid tokens
    valid_tokens = padded_inputs[:, seq_len:current_pos].view(-1).tolist()
    answer = tokenizer.decode(valid_tokens, skip_special_tokens=True)

    return answer


def download_model_and_tokenizer(model_name, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct", trust_remote_code=True)

    # Prepare input sentence
    messages = [{"role": "user", "content": "write a bubble sort algorithm in python."}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    return model, tokenizer, inputs


class DeepSeekWrapper(torch.nn.Module):
    def __init__(self, model, max_new_tokens=200):
        super().__init__()
        self.model = model
        self.max_new_tokens = max_new_tokens

    def forward(self, input_tensor):
        return self.model(input_tensor, max_new_tokens=self.max_new_tokens).logits
