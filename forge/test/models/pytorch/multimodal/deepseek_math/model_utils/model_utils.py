# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def generation(max_new_tokens, compiled_model, input_ids, tokenizer):
    for i in range(max_new_tokens):
        logits = compiled_model(input_ids)
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text


def download_model_and_tokenizer(model_name, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")

    model.generation_config = GenerationConfig.from_pretrained(model_name)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    model.generation_config.use_cache = kwargs.get("use_cache", False)

    # Prepare input sentence
    messages = kwargs.get(
        "messages",
        [
            {
                "role": "user",
                "content": "what is the integral of x^2 from 0 to 2?\nPlease reason step by step, and put your final answer within \\boxed{}.",
            }
        ],
    )
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

    return model, tokenizer, input_ids


class DeepSeekWrapper(torch.nn.Module):
    def __init__(self, model, max_new_tokens=200):
        super().__init__()
        self.model = model
        self.max_new_tokens = max_new_tokens

    def forward(self, input_tensor):
        return self.model(input_tensor, max_new_tokens=self.max_new_tokens).logits


class DeepSeekWrapper_decoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_tensor):
        output = self.model(input_tensor)
        return output.last_hidden_state
