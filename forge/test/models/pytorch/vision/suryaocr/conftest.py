# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import sys
from importlib.abc import MetaPathFinder
from importlib.util import spec_from_loader
from typing import Optional

import torch


class SuryaLoaderFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "surya.recognition.loader":
            return spec_from_loader(fullname, SuryaLoader())


class SuryaLoader:
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        from surya.common.load import ModelLoader
        from surya.recognition.model.config import (
            DonutSwinConfig,
            SuryaOCRConfig,
            SuryaOCRDecoderConfig,
            SuryaOCRTextEncoderConfig,
        )
        from surya.recognition.model.encoderdecoder import OCREncoderDecoderModel
        from surya.recognition.processor import SuryaProcessor
        from surya.settings import settings

        class RecognitionModelLoader(ModelLoader):
            def __init__(self, checkpoint: Optional[str] = None):
                super().__init__(checkpoint)
                if self.checkpoint is None:
                    self.checkpoint = settings.RECOGNITION_MODEL_CHECKPOINT

            def model(self, device="cpu", dtype=torch.float32) -> OCREncoderDecoderModel:
                config = SuryaOCRConfig.from_pretrained(self.checkpoint)
                config.decoder = SuryaOCRDecoderConfig(**config.decoder)
                config.encoder = DonutSwinConfig(**config.encoder)
                config.text_encoder = SuryaOCRTextEncoderConfig(**config.text_encoder)

                model = OCREncoderDecoderModel.from_pretrained(self.checkpoint, config=config, torch_dtype=dtype)
                model = model.to(device).eval()

                return model

            def processor(self) -> SuryaProcessor:
                return SuryaProcessor(self.checkpoint)

        module.RecognitionModelLoader = RecognitionModelLoader


sys.meta_path.insert(0, SuryaLoaderFinder())
