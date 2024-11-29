# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask


def generate_fuyu_embedding(model, input_ids, image_patches, image_patches_indices):
    inputs_embeds = model.language_model.get_input_embeddings()(input_ids)
    patch_embeddings = model.vision_embed_tokens(image_patches.to(model.vision_embed_tokens.weight.dtype))
    inputs_embeds = model.gather_continuous_embeddings(
        word_embeddings=inputs_embeds,
        continuous_embeddings=patch_embeddings,
        image_patch_input_indices=image_patches_indices,
    )
    return inputs_embeds


class FuyuModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.fuyu_model = model
        self.fuyu_config = model.config

    def forward(self, inputs_embeds):
        output_attentions = self.fuyu_config.output_attentions
        use_cache = self.fuyu_config.use_cache

        # retrieve input_ids and inputs_embeds
        batch_size, seq_length, _ = inputs_embeds.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        # device = input_ids.device if input_ids is not None else inputs_embeds.device
        device = inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)

        # PersimmonForCausalLM
        output_hidden_states = self.fuyu_model.language_model.config.output_hidden_states

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.fuyu_model.language_model.model(
            input_ids=None,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )

        return outputs


class FuyuModelImgDecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.fuyu_model = model
        self.fuyu_config = model.config

    def forward(self, inputs_embeds, attention_mask):
        batch_size, seq_length, hidden_dim = inputs_embeds.shape
        position_ids = torch.arange(seq_length, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        hidden_states = inputs_embeds

        presents = []
        for idx, decoder_layer in enumerate(self.fuyu_model.language_model.model.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=False,
                use_cache=True,
            )

            hidden_states = layer_outputs[0]
            presents.append(layer_outputs[1])

        hidden_states = self.fuyu_model.language_model.model.final_layernorm(hidden_states)
        return hidden_states, *presents


class FuyuModelTxtDecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.fuyu_model = model
        self.fuyu_config = model.config

    def forward(self, inputs_embeds, attention_mask, position_ids, *past_key_values):
        batch_size, seq_length, _ = inputs_embeds.shape
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values[0].shape[-2]
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        hidden_states = inputs_embeds

        presents = []
        for idx, decoder_layer in enumerate(self.fuyu_model.language_model.model.layers):
            pkv = tuple([past_key_values[(idx * 2) + j] for j in range(2)])

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=pkv,
                output_attentions=False,
                use_cache=True,
            )

            hidden_states = layer_outputs[0]
            presents.append(layer_outputs[1])

        hidden_states = self.fuyu_model.language_model.model.final_layernorm(hidden_states)
        return hidden_states, *presents
