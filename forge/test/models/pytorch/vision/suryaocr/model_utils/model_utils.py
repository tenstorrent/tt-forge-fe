# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import json
import os
from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from loguru import logger
from PIL import Image
from surya.debug.text import draw_text_on_image
from surya.detection import DetectionPredictor
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor


class SuryaOCRWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.detection_predictor = DetectionPredictor()
        self.foundation_predictor = FoundationPredictor()
        self.rec_predictor = RecognitionPredictor(self.foundation_predictor)
        self._to_pil = transforms.ToPILImage()

        # Set eval mode on wrapper and underlying models
        self.eval()
        if hasattr(self.rec_predictor, "model"):
            self.rec_predictor.model.eval()
        if hasattr(self.detection_predictor, "model"):
            self.detection_predictor.model.eval()
        if hasattr(self.foundation_predictor, "model"):
            self.foundation_predictor.model.eval()

        # Force eager attention to avoid CPU SDPA/flash path during Forge tracing
        for predictor in [self.rec_predictor, self.detection_predictor, self.foundation_predictor]:
            model = getattr(predictor, "model", None)
            if model is None:
                continue
            for comp in [
                getattr(model, "decoder", None),
                getattr(model, "vision_encoder", None),
                getattr(model, "config", None),
            ]:
                if comp is not None and hasattr(comp, "_attn_implementation"):
                    try:
                        comp._attn_implementation = "eager"
                    except Exception:
                        pass

    def forward(self, images_tensor: torch.Tensor):
        batch_size = images_tensor.shape[0]
        images: List[Image.Image] = [self._to_pil(images_tensor[i].cpu()) for i in range(batch_size)]
        highres_images: List[Image.Image] = images
        task_names = ["ocr_with_boxes"] * len(images)
        predictions_by_image = self.rec_predictor(
            images,
            task_names=task_names,
            det_predictor=self.detection_predictor,
            highres_images=highres_images,
        )
        # Pack to tensors
        lines_bbox, lines_conf, text_codes, text_len, lines_len = pack_predictions(predictions_by_image)
        # Tie outputs to input to avoid constant folding in generated module
        zero_f = images_tensor.sum().to(lines_bbox.dtype) * 0
        zero_i = images_tensor.sum().to(text_codes.dtype) * 0
        lines_bbox = lines_bbox + zero_f
        lines_conf = lines_conf + zero_f
        text_codes = text_codes + zero_i
        text_len = text_len + zero_i
        lines_len = lines_len + zero_i
        return lines_bbox, lines_conf, text_codes, text_len, lines_len


class TextLineLite:
    __slots__ = ("text", "bbox", "confidence")

    def __init__(self, text, bbox, confidence):
        self.text = text
        self.bbox = bbox
        self.confidence = confidence


class OCRResultLite:
    __slots__ = ("text_lines",)

    def __init__(self, text_lines):
        self.text_lines = text_lines


def pack_predictions(preds, max_lines=50000, max_chars=50000):
    B = len(preds)
    lines_bbox = torch.zeros(B, max_lines, 4, dtype=torch.float32)
    lines_conf = torch.zeros(B, max_lines, dtype=torch.float32)
    text_codes = torch.full((B, max_lines, max_chars), fill_value=-1, dtype=torch.int32)
    text_len = torch.zeros(B, max_lines, dtype=torch.int32)
    lines_len = torch.zeros(B, dtype=torch.int32)

    for b, p in enumerate(preds):
        lines = getattr(p, "text_lines", [])[:max_lines]
        lines_len[b] = len(lines)
        for i, line in enumerate(lines):
            if hasattr(line, "bbox") and line.bbox is not None:
                lines_bbox[b, i] = torch.tensor(line.bbox, dtype=torch.float32)
            if hasattr(line, "confidence"):
                lines_conf[b, i] = float(line.confidence)
            t = getattr(line, "text", "") or ""
            codes = [ord(c) for c in t][:max_chars]
            if len(codes) > 0:
                text_codes[b, i, : len(codes)] = torch.tensor(codes, dtype=torch.int32)
            text_len[b, i] = len(codes)

    return lines_bbox, lines_conf, text_codes, text_len, lines_len


def unpack_predictions(lines_bbox, lines_conf, text_codes, text_len, lines_len):
    B, K, _ = lines_bbox.shape
    results = []
    for b in range(B):
        num = int(lines_len[b].item())
        page_lines = []
        for i in range(num):
            L = int(text_len[b, i].item())
            codes = text_codes[b, i, :L].tolist()
            text = "".join(chr(c) for c in codes)
            bbox = lines_bbox[b, i].tolist()
            conf = float(lines_conf[b, i].item())
            page_lines.append({"text": text, "bbox": bbox, "confidence": conf})
        results.append({"text_lines": page_lines})
    return results


def freeze_all(wrapper, warmup_input: torch.Tensor = None):
    """Warm up to instantiate any lazy modules, then freeze all parameters found under predictor `.model` modules."""
    import torch.nn as nn

    # Warmup forward to trigger any lazy construction inside predictors
    if warmup_input is not None:
        try:
            with torch.inference_mode():
                _ = wrapper(warmup_input)
        except Exception:
            pass

    def freeze_module(m: nn.Module):
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    # Freeze registered submodules off the wrapper itself
    for _, m in wrapper.named_modules():
        if isinstance(m, nn.Module):
            for p in m.parameters():
                p.requires_grad = False

    # Freeze predictor `.model` modules explicitly
    for obj in [
        getattr(wrapper, "rec_predictor", None),
        getattr(wrapper, "detection_predictor", None),
        getattr(wrapper, "foundation_predictor", None),
    ]:
        model_attr = getattr(obj, "model", None) if obj is not None else None
        if isinstance(model_attr, nn.Module):
            freeze_module(model_attr)


def dicts_to_objects(reconstructed):
    results = []
    for page in reconstructed:
        tls = [TextLineLite(line["text"], line["bbox"], line["confidence"]) for line in page["text_lines"]]
        results.append(OCRResultLite(tls))
    return results


def save_outputs(co_out, images):
    names: List[str] = ["excerpt_text"]
    lines_bbox, lines_conf, text_codes, text_len, lines_len = co_out
    reconstructed = unpack_predictions(lines_bbox, lines_conf, text_codes, text_len, lines_len)

    # Convert dicts to lightweight objects with attributes
    predictions_by_image = dicts_to_objects(reconstructed)

    # Prepare output directory
    base_out = os.path.join(os.getcwd(), "surya_results")
    result_path = os.path.abspath(os.path.join(base_out, names[0]))
    result_path = "/proj_sw/user_dev/mramanathan/bgdlab14_aug18_forge_new/tt-forge-fe/forge/test/models/pytorch/vision/suryaocr/surya_out_latest"
    os.makedirs(result_path, exist_ok=True)

    # Save visualization PNGs
    for idx, (name, image, pred) in enumerate(zip(names, images, predictions_by_image)):
        bboxes = [line.bbox for line in pred.text_lines]
        pred_text = [line.text for line in pred.text_lines]
        page_image = draw_text_on_image(bboxes, pred_text, image.size)
        page_image.save(os.path.join(result_path, f"{name}_{idx}_text.png"))

    # Write results.json
    out_preds = defaultdict(list)
    for name, pred, image in zip(names, predictions_by_image, images):
        page_dict = {
            "text_lines": [{"text": tl.text, "bbox": tl.bbox, "confidence": tl.confidence} for tl in pred.text_lines]
        }
        page_dict["page"] = len(out_preds[name]) + 1
        out_preds[name].append(page_dict)

    with open(os.path.join(result_path, "results.json"), "w+", encoding="utf-8") as f:
        json.dump(out_preds, f, ensure_ascii=False)

    logger.info(f"Wrote results to {result_path}")
