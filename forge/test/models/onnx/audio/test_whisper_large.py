# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from datasets import load_dataset
from transformers import AutoFeatureExtractor, WhisperModel

import forge
from forge.forge_property_utils import Framework, ModelArch, Source, Task, record_model_properties
from forge.verify.verify import verify

from test.utils import download_model
import onnx


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_features, decoder_input_ids):
        inputs = {"input_features": input_features, "decoder_input_ids": decoder_input_ids}
        output = self.model(**inputs)
        return output


variants = ["openai/whisper-large-v3"]


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_whisper_large_v3_onnx(variant, tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.WHISPER,
        variant=variant,
        task=Task.AUDIO_ASR,
        source=Source.HUGGINGFACE,
    )

    pytest.xfail(reason="Requires multi-chip support")

    # Load Model and feature extractor
    model = download_model(WhisperModel.from_pretrained, variant, return_dict=False)
    torch_model = Wrapper(model)
    feature_extractor = download_model(AutoFeatureExtractor.from_pretrained, variant)

    # prepare input
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    input_audio = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
    input_features = input_audio.input_features
    decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
    inputs = [input_features, decoder_input_ids]

    # Export model to ONNX
    onnx_path = f"{tmp_path}/whisper_v3.onnx"
    torch.onnx.export(torch_model, (inputs[0], inputs[1]), onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)

    # passing model file instead of model proto due to size of the model(>2GB) - #https://github.com/onnx/onnx/issues/3775#issuecomment-943416925
    onnx.checker.check_model(onnx_path)
    framework_model = forge.OnnxModule(module_name, onnx_model, onnx_path)

    # Compile model
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    # Model Verification and inference
    _, cout = verify(
        inputs,
        framework_model,
        compiled_model,
    )
