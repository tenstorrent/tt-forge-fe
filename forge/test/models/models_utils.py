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
from typing import List
from surya.common.util import mark_step
from tqdm import tqdm
from surya.settings import settings
from surya.detection import DetectionPredictor
from surya.input.processing import convert_if_not_rgb
from surya.recognition.schema import OCRResult
import torch.nn.functional as F

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


def __call__without_post_processing(
    self,
    images: List[Image.Image],
    langs: List[List[str] | None],
    det_predictor: DetectionPredictor | None = None,
    detection_batch_size: int | None = None,
    recognition_batch_size: int | None = None,
    highres_images: List[Image.Image] | None = None,
    bboxes: List[List[List[int]]] | None = None,
    polygons: List[List[List[List[int]]]] | None = None,
    sort_lines: bool = True,
) -> List[OCRResult]:
    assert len(images) == len(langs), "You need to pass in one list of languages for each image"
    images = convert_if_not_rgb(images)
    if highres_images is not None:
        assert len(images) == len(highres_images), "You need to pass in one highres image for each image"

    highres_images = convert_if_not_rgb(highres_images) if highres_images is not None else [None] * len(images)

    if bboxes is None and polygons is None:
        assert (
            det_predictor is not None
        ), "You need to pass in a detection predictor if you don't provide bboxes or polygons"

        # Detect then slice
        flat = self.detect_and_slice_bboxes(
            images, langs, det_predictor, detection_batch_size=detection_batch_size, highres_images=highres_images
        )
    else:
        if bboxes is not None:
            assert len(images) == len(bboxes), "You need to pass in one list of bboxes for each image"
        if polygons is not None:
            assert len(images) == len(polygons), "You need to pass in one list of polygons for each image"

        flat = self.slice_bboxes(images, langs, bboxes=bboxes, polygons=polygons)

    output = self.batch_recognition(flat["slices"], flat["langs"], batch_size=recognition_batch_size)

    return output


def batch_recognition_without_post_processing(
    self, images: List[Image.Image], languages: List[List[str] | None], batch_size=None
):
    assert all(isinstance(image, Image.Image) for image in images)
    assert len(images) == len(languages)

    if len(images) == 0:
        return [], []

    if batch_size is None:
        batch_size = self.get_batch_size()

    # Sort images by width, so similar length ones go together
    sorted_pairs = sorted(enumerate(images), key=lambda x: x[1].width, reverse=False)
    indices, images = zip(*sorted_pairs)
    indices = list(indices)
    images = list(images)

    batch_predictions_all = []
    sequence_scores_all = []

    for i in tqdm(range(0, len(images), batch_size), desc="Recognizing Text", disable=self.disable_tqdm):
        batch_images = images[i : i + batch_size]
        batch_images = [image.convert("RGB") for image in batch_images]  # also copies the images
        current_batch_size = len(batch_images)
        batch_langs = languages[i : i + current_batch_size]
        processed_batch = self.processor(text=[""] * len(batch_images), images=batch_images, langs=batch_langs)

        batch_pixel_values = processed_batch["pixel_values"]
        batch_langs = processed_batch["langs"]
        batch_pixel_values, batch_decoder_input = self.prepare_input(batch_langs, batch_pixel_values, batch_size)

        token_count = 0
        inference_token_count = batch_decoder_input.shape[-1]

        decoder_position_ids = (
            torch.ones_like(batch_decoder_input[0, :], dtype=torch.int64, device=self.model.device).cumsum(0) - 1
        )
        self.model.decoder.model._setup_cache(self.model.config, batch_size, self.model.device, self.model.dtype)
        self.model.text_encoder.model._setup_cache(self.model.config, batch_size, self.model.device, self.model.dtype)

        # Batch pixel values is the real current batch size
        sequence_scores = torch.zeros(
            batch_pixel_values.shape[0], dtype=torch.bool, device=self.model.device
        ).unsqueeze(1)
        all_done = torch.zeros(batch_pixel_values.shape[0], dtype=torch.bool, device=self.model.device)
        batch_predictions = torch.zeros(
            batch_pixel_values.shape[0], dtype=torch.int64, device=self.model.device
        ).unsqueeze(1)
        device_pad_token = torch.tensor(self.processor.tokenizer.pad_token_id, device=self.model.device)

        with settings.INFERENCE_MODE():
            encoder_hidden_states = self.model.encoder(pixel_values=batch_pixel_values).last_hidden_state

            text_encoder_input_ids = (
                torch.arange(
                    self.model.text_encoder.config.query_token_count,
                    device=encoder_hidden_states.device,
                    dtype=torch.long,
                )
                .unsqueeze(0)
                .expand(encoder_hidden_states.size(0), -1)
            )

            encoder_text_hidden_states = self.model.text_encoder(
                input_ids=text_encoder_input_ids,
                cache_position=None,
                attention_mask=None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=None,
                use_cache=False,
            ).hidden_states

            while token_count < settings.RECOGNITION_MAX_TOKENS - 1:
                is_prefill = token_count == 0
                # TODO: add attention mask
                return_dict = self.model.decoder(
                    input_ids=batch_decoder_input,
                    encoder_hidden_states=encoder_text_hidden_states,
                    cache_position=decoder_position_ids,
                    use_cache=True,
                    prefill=is_prefill,
                )

                decoder_position_ids = decoder_position_ids[-1:] + 1
                logits = return_dict["logits"]  # Ignore batch padding

                preds = torch.argmax(logits[:, -1], dim=-1)
                scores = torch.max(F.softmax(logits[:, -1], dim=-1), dim=-1).values.unsqueeze(1)
                done = (preds == self.processor.tokenizer.eos_id) | (preds == self.processor.tokenizer.pad_id)
                all_done = all_done | done
                all_done_cpu = all_done.cpu()

                # Confidence score for the current token
                scores = scores.masked_fill(all_done, 0)
                sequence_scores = torch.cat([sequence_scores, scores], dim=1)

                # Account for possible padding
                if all_done_cpu[:current_batch_size].all():
                    break

                batch_decoder_input = preds.unsqueeze(1)

                # If this batch item is done, input a pad token
                batch_decoder_input = torch.where(all_done.unsqueeze(1), device_pad_token, batch_decoder_input)

                batch_predictions = torch.cat([batch_predictions, batch_decoder_input], dim=1)
                token_count += inference_token_count

                inference_token_count = batch_decoder_input.shape[-1]
                mark_step()

        sequence_scores = torch.sum(sequence_scores, dim=-1) / torch.sum(sequence_scores != 0, dim=-1)

        sequence_scores = sequence_scores.cpu()[:current_batch_size]
        batch_predictions = batch_predictions.cpu()[:current_batch_size, 1:]  # Remove the start token

        print("batch_predictions", batch_predictions)
        print("sequence_scores", sequence_scores)

        batch_predictions_all.append(batch_predictions)
        sequence_scores_all.append(sequence_scores)

    return (*batch_predictions_all, *sequence_scores_all)
