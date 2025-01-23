# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Whisper Demo - Conditional Generation
# Example of ASR pipeline: https://github.com/huggingface/transformers/blob/ae54e3c3b18bac0832ad62ea9b896dfd52a09850/tests/pipelines/test_pipelines_automatic_speech_recognition.py#L695

import copy
import os
import time

import pytest
import torch
from transformers import (
    AutoProcessor,
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    pipeline,
)

import forge
from forge.config import _get_global_compiler_config
from forge.forgeglobal import TILE_DIM
from forge.transformers.pipeline import pipeline as forge_pipeline
from forge.verify.verify import verify

from test.models.utils import Framework, build_module_name
from test.utils import download_model

variants = [
    "openai/whisper-tiny",
    "openai/whisper-base",
    "openai/whisper-small",
    "openai/whisper-medium",
    "openai/whisper-large",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_whisper(record_forge_property, variant):
    if variant != "openai/whisper-tiny":
        pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="whisper", variant=variant)

    # Record Forge Property
    record_forge_property("model_name", module_name)

    # Load model (with tokenizer and feature extractor)
    processor = download_model(AutoProcessor.from_pretrained, variant)
    model_config = WhisperConfig.from_pretrained(variant)
    model = download_model(
        WhisperForConditionalGeneration.from_pretrained,
        variant,
        config=model_config,
    )
    model.config.use_cache = False

    # Load and preprocess sample audio
    sample = torch.load("forge/test/models/files/samples/audio/1272-128104-0000.pt")
    sample_audio = sample["audio"]["array"]

    inputs = processor(sample_audio, return_tensors="pt")
    input_features = inputs.input_features

    # Get decoder inputs
    decoder_start_token_tensor = torch.tensor(model.generation_config.decoder_start_token_id, dtype=torch.long)
    decoder_input_ids = torch.ones((1, 1), dtype=torch.long) * decoder_start_token_tensor

    inputs = [input_features, decoder_input_ids]

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_features, decoder_input_ids):
            inputs = {"input_features": input_features, "decoder_input_ids": decoder_input_ids}
            output = self.model(**inputs)
            return output.logits

    framework_model = Wrapper(model)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)

    current_decoder_input_ids = decoder_input_ids
    all_decoded_ids = decoder_input_ids

    # The iteration count in for _ in range(1) is deliberately limited to 1 to prevent shape mismatches.
    # The model has been compiled specifically for the first decoding step, where decoder_input_ids
    # has a fixed length of (1,1) (the initial token). However, in generative models like Whisper, the length of
    # decoder_input_ids increases with each decoding step as tokens are appended to the sequence.
    # This dynamic increase in shape is incompatible with the static shape expected by the compiled model,
    # leading to a runtime error if subsequent iterations are attempted.

    for _ in range(1):

        # Inference
        outputs = compiled_model(input_features, current_decoder_input_ids)
        logits = outputs[0]

        # Get the next token ID (greedy decoding)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)

        # Break if EOS token is generated
        if next_token.item() == model.generation_config.eos_token_id:
            break

        # Append next token to sequence
        all_decoded_ids = torch.cat([all_decoded_ids, next_token], dim=-1)

        # Update decoder inputs for the next iteration
        current_decoder_input_ids = all_decoded_ids

    print("summary : ", processor.decode(all_decoded_ids[0], skip_special_tokens=True))


@pytest.mark.skip_model_analysis
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.skip(reason="Redundant")
def test_whisper_pipeline(record_forge_property, variant):
    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="whisper", variant=variant, suffix="pipeline")

    # Record Forge Property
    record_forge_property("model_name", module_name)

    # Configurations
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.amp_level = 2
    compiler_cfg.enable_link_past_cache_ios = False
    compiler_cfg.enable_tvm_cpu_fallback = False  # Run full model on silicon
    compiler_cfg.enable_enumerate_u_kt = False
    compiler_cfg.default_df_override = forge._C.DataFormat.Float16_b

    # Load model (with tokenizer and feature extractor)
    framework_model = download_model(WhisperForConditionalGeneration.from_pretrained, variant)
    tokenizer = download_model(WhisperTokenizer.from_pretrained, variant)
    feature_extractor = download_model(WhisperFeatureExtractor.from_pretrained, variant)

    ### Load HF pipeline
    hf_pipeline = pipeline(
        "automatic-speech-recognition",
        model=copy.deepcopy(framework_model),
        tokenizer=copy.deepcopy(tokenizer),
        feature_extractor=copy.deepcopy(feature_extractor),
    )

    ### Load Forge pipeline
    asr_pipeline = forge_pipeline(
        "automatic-speech-recognition",
        model=framework_model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        forge_max_length=32,
    )

    # Load & preprocess sample audio
    ### Load from HF datasets & preprocess
    # data_set = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    # sample = processor(data_set[0]["audio"]["array"], return_tensors="pt")
    ### Load preprocessed from local
    sample = torch.load("forge/test/models/files/samples/audio/1272-128104-0000.pt")
    sample_audio = sample["audio"]["array"]
    ### Load direct audio file
    # sample_audio = "audio_demos/whisper/data_sample/1272-128104-0000.flac"

    # Sanity run on CPU
    cpu_out = hf_pipeline(sample_audio)
    print("HF pipeline:", cpu_out["text"])

    # Compile and run on TT device
    tt_out = asr_pipeline(sample_audio)
    print("TT pipeline:", tt_out["text"])

    # Compare outputs
    assert cpu_out["text"] == tt_out["text"]


@pytest.mark.skip_model_analysis
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.skip(reason="Not supported")
def test_whisper_encoder(record_forge_property, test_device, variant):
    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="whisper", variant=variant, suffix="encoder")

    # Record Forge Property
    record_forge_property("model_name", module_name)

    # Configurations
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_cpu_fallback = False
    compiler_cfg.input_queues_on_host = True
    compiler_cfg.enable_link_past_cache_ios = True
    os.environ["FORGE_FORCE_SEQUENTIAL"] = "1"

    config = WhisperConfig.from_pretrained(variant)
    config.return_dict = False
    config.encoder_layers = 1
    config.decoder_layers = 1
    pad_model = True
    if pad_model:
        pad_to_tiles = 48
        padded_len = pad_to_tiles * TILE_DIM
        pad_amount = padded_len - config.max_source_positions
        config.max_source_positions = padded_len

    model = download_model(
        WhisperForConditionalGeneration.from_pretrained,
        variant,
        ignore_mismatched_sizes=True,
        config=config,
    )
    if pad_model:
        unpadded_model = WhisperForConditionalGeneration.from_pretrained(variant)
        padded_param = torch.nn.functional.pad(
            unpadded_model.model.encoder.embed_positions.weight.data, (0, 0, 0, pad_amount)
        )
        model.model.encoder.embed_positions.weight.data = padded_param

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_features):
            enc_out = self.model.model.encoder(input_features)

            return enc_out[0]

    # Load model (with tokenizer and feature extractor)
    processor = download_model(AutoProcessor.from_pretrained, variant)

    model = Wrapper(model)
    forge_model = PyTorchModule("pt_whisper", model)

    # Load and preprocess sample audio
    sample = torch.load("forge/test/models/files/samples/audio/1272-128104-0000.pt")
    sample_audio = sample["audio"]["array"]

    inputs = processor(sample_audio, return_tensors="pt")
    if pad_model:
        input_features = torch.nn.functional.pad(inputs.input_features, (0, pad_amount * 2, 0, 0))
    else:
        input_features = inputs.input_features

    tt0 = forge.TTDevice("tt0", devtype=test_device.devtype, arch=test_device.arch, module=forge_model)
    output_q = forge.initialize_pipeline(
        training=False,
        sample_inputs=(input_features,),
    )

    start = time.time()
    tokens_to_generate = 10 if test_device.devtype == BackendType.Silicon else 3
    for _ in range(tokens_to_generate):
        tt0.push_to_inputs(input_features)
        forge.run_forward(input_count=1)
        ans = output_q.get()

    end = time.time()
    print(
        f"Ran {tokens_to_generate} iterations. Duration: {(end - start) / tokens_to_generate} seconds, speed: {(tokens_to_generate) / (end - start)} iters/sec"
    )
