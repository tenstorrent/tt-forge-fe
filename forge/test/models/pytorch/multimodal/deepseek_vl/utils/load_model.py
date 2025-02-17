# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from transformers import AutoModelForCausalLM

from forge.verify.verify import verify

from test.models.pytorch.multimodal.deepseek_vl.utils.models.modeling_vlm import (
    MultiModalityCausalLM,
    VLChatProcessor,
    load_pil_images,
)


def verify_deepseek_vl(inputs_embeds, framework_model, compiled_model, max_new_tokens=512):
    batch_size, seq_len, embed_dim = inputs_embeds.shape
    max_seq_len = seq_len + max_new_tokens  # Fixed total sequence length

    padded_inputs_embeds = torch.zeros(
        (batch_size, max_seq_len, embed_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device
    )
    padded_inputs_embeds[:, :seq_len, :] = inputs_embeds  # Copy initial embeddings
    verify([padded_inputs_embeds], framework_model, compiled_model)


def generate_model_deepseek_vl_pytorch(variant):

    model_path = variant
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, model, max_new_tokens=512):
            super().__init__()
            self.model = model
            self.eos_token_id = tokenizer.eos_token_id
            self.bos_token_id = tokenizer.bos_token_id
            self.pad_token_id = tokenizer.pad_token_id
            self.max_new_tokens = max_new_tokens

        def forward(self, inputs_embeds):
            return self.model.language_model(
                inputs_embeds=inputs_embeds,
                pad_token_id=self.pad_token_id,
                bos_token_id=self.bos_token_id,
                eos_token_id=self.eos_token_id,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=False,
            ).logits

    framework_model = Wrapper(vl_gpt)

    # Single image conversation example
    conversation = [
        {
            "role": "User",
            "content": "<image_placeholder>Describe each stage of this image.",
            "images": ["forge/test/models/pytorch/multimodal/deepseek_vl/image/training_pipelines.jpg"],
        },
        {"role": "Assistant", "content": ""},
    ]

    # Load images and prepare for inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(conversations=conversation, images=pil_images, force_batchify=True)

    # Run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    return framework_model, vl_gpt, tokenizer, inputs_embeds


def generate(max_new_tokens, model, inputs_embeds, tokenizer, vl_gpt):
    batch_size, seq_len, embed_dim = inputs_embeds.shape
    max_seq_len = seq_len + max_new_tokens  # Fixed total sequence length

    padded_inputs_embeds = torch.zeros(
        (batch_size, max_seq_len, embed_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device
    )
    padded_inputs_embeds[:, :seq_len, :] = inputs_embeds  # Copy initial embeddings

    generated_token_ids = torch.full(
        (batch_size, max_seq_len), tokenizer.eos_token_id, dtype=torch.long, device=vl_gpt.device
    )
    current_pos = seq_len

    for _ in range(max_new_tokens):
        logits = model(padded_inputs_embeds)

        # Get only the logits corresponding to the last valid token
        if isinstance(logits, list):
            next_token_logits = logits[0][:, current_pos - 1, :]
        else:
            next_token_logits = logits[:, current_pos - 1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)

        # Stop if EOS token is encountered
        if next_token_id.item() == tokenizer.eos_token_id:
            break

        # Store generated token
        generated_token_ids[:, current_pos] = next_token_id

        # Update embeddings in fixed position
        new_embedding = vl_gpt.language_model.get_input_embeddings()(next_token_id.unsqueeze(0))
        padded_inputs_embeds[:, current_pos, :] = new_embedding.squeeze(0)

        current_pos += 1  # Move to next position

    # Decode valid tokens
    valid_tokens = generated_token_ids[:, seq_len:current_pos].view(-1).tolist()
    answer = tokenizer.decode(valid_tokens, skip_special_tokens=True)

    return answer
