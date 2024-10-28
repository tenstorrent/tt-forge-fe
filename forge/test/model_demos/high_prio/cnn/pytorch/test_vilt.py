# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
import os
import forge
import requests
import torch
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering, ViltForMaskedLM, ViltConfig

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

text1 = "How many cats are there?"
text2 = "a bunch of cats laying on a [MASK]."


class ViLtEmbeddingWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vilt_model = model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        pixel_values=None,
        pixel_mask=None,
        inputs_embeds=None,
        image_embeds=None,
        image_token_type_idx=None,
    ):

        embeddings, masks = self.vilt_model.vilt.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            image_token_type_idx=image_token_type_idx,
        )
        return embeddings, masks


class ViltModelWrapper(torch.nn.Module):
    def __init__(self, model, task=None, text_seq_len=None):
        super().__init__()
        self.vilt_model = model
        self.task = task
        self.text_seq_len = text_seq_len

    def forward(self, embedding_output, attention_mask, head_mask=None):

        head_mask = self.vilt_model.vilt.get_head_mask(head_mask, self.vilt_model.vilt.config.num_hidden_layers)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float32).min

        encoder_outputs = self.vilt_model.vilt.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            return_dict=False,
        )

        sequence_output = encoder_outputs[0]

        sequence_output = self.vilt_model.vilt.layernorm(sequence_output)
        pooled_output = (
            self.vilt_model.vilt.pooler(sequence_output) if self.vilt_model.vilt.pooler is not None else None
        )

        viltmodel_output = (sequence_output, pooled_output) + encoder_outputs[1:]

        sequence_output, pooled_output = viltmodel_output[:2]

        if self.task == "maskedlm":

            if self.text_seq_len is None:
                raise ValueError("You cannot must provide text sequence length")

            text_features, _ = (sequence_output[:, : self.text_seq_len], sequence_output[:, self.text_seq_len :])

            mlm_logits = self.vilt_model.mlm_score(text_features)

            viltmodel_output = (mlm_logits,) + viltmodel_output[2:]

        if self.task == "qa":

            logits = self.vilt_model.classifier(pooled_output)

            viltmodel_output = (logits,) + viltmodel_output[2:]

        return viltmodel_output


def generate_model_vilt_question_answering_hf_pytorch(test_device, variant):

    # Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH
    os.environ["FORGE_DISABLE_ERASE_INVERSE_OPS_PASS"] = "1"

    # Set model configurations
    config = ViltConfig.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config = ViltConfig(**config_dict)

    # Load model and processor from HuggingFace
    processor = download_model(ViltProcessor.from_pretrained, variant)
    model = download_model(ViltForQuestionAnswering.from_pretrained, variant, config=config)
    model.eval()

    encoding = processor(image, text1, return_tensors="pt")

    # Wrapper
    text_vision_embedding_model = ViLtEmbeddingWrapper(model)
    vilt_model = ViltModelWrapper(model, task="qa")

    embedding_output, attention_mask = text_vision_embedding_model(**encoding)

    return vilt_model, [embedding_output.detach().cpu(), attention_mask.detach().cpu().to(torch.float32)], {}


variants = ["dandelin/vilt-b32-finetuned-vqa"]


@pytest.mark.parametrize("variant", variants, ids=variants)
def test_vilt_question_answering_hf_pytorch(variant, test_device):
    model, inputs, _ = generate_model_vilt_question_answering_hf_pytorch(
        test_device,
        variant,
    )
    compiled_model = forge.compile(
        model, sample_inputs=[inputs[0], inputs[1]], module_name="pt_ViLt_question_answering"
    )


def generate_model_vilt_maskedlm_hf_pytorch(test_device, variant):

    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH
    os.environ["FORGE_DISABLE_ERASE_INVERSE_OPS_PASS"] = "1"

    # Set model configurations
    config = ViltConfig.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config = ViltConfig(**config_dict)

    # Load model and processor from HuggingFace
    processor = download_model(ViltProcessor.from_pretrained, variant)
    model = download_model(ViltForMaskedLM.from_pretrained, variant, config=config)
    model.eval()

    # prepare inputs
    encoding = processor(image, text2, return_tensors="pt")

    # Wrapper
    text_vision_embedding_model = ViLtEmbeddingWrapper(model)
    vilt_model = ViltModelWrapper(model=model, task="maskedlm", text_seq_len=encoding["input_ids"].shape[1])

    embedding_output, attention_mask = text_vision_embedding_model(**encoding)

    return vilt_model, [embedding_output.detach().cpu(), attention_mask.detach().cpu().to(torch.float32)], {}


variants = ["dandelin/vilt-b32-mlm"]


@pytest.mark.parametrize("variant", variants, ids=variants)
def test_vilt_maskedlm_hf_pytorch(variant, test_device):
    model, inputs, _ = generate_model_vilt_maskedlm_hf_pytorch(
        test_device,
        variant,
    )
    compiled_model = forge.compile(model, sample_inputs=[inputs[0], inputs[1]], module_name="pt_ViLt_maskedlm")
