# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask


# SPDX-FileCopyrightText: Copyright (c) 2021 OpenAI
#
# SPDX-License-Identifier: MIT
# https://github.com/openai/CLIP
class CLIPVisionWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.clip_model = model

    def forward(self, pixel_values):

        vision_outputs = self.clip_model.vision_model(pixel_values, return_dict=False)
        return vision_outputs


class CLIPTextWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.clip_model = model

    def forward(self, input_ids, attention_mask):

        # text_outputs = self.clip_model.text_model(input_ids, attention_mask, return_dict=False)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.clip_model.text_model.embeddings(input_ids=input_ids, position_ids=None)

        bsz, seq_len = input_shape
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.clip_model.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            return_dict=False,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.clip_model.text_model.final_layer_norm(last_hidden_state)

        return (last_hidden_state, *encoder_outputs)


class CLIPPostProcessingWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.clip_model = model

    def forward(self, input_ids, vision_outputs, last_hidden_state, *encoder_outputs):
        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        ]

        text_outputs = (last_hidden_state, pooled_output) + encoder_outputs[1:]

        image_embeds = vision_outputs[1]
        image_embeds = self.clip_model.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.clip_model.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        output = (logits_per_image, logits_per_text, text_embeds, image_embeds, *text_outputs, *vision_outputs)
        return output
