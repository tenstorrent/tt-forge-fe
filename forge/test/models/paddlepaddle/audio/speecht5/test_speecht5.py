# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from paddlenlp.transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5ForSpeechToText,
    SpeechT5ForSpeechToSpeech,
)
import paddle
import pytest
from datasets import load_dataset

import forge
from forge.verify.verify import verify
from forge.tvm_calls.forge_utils import paddle_trace

from forge.forge_property_utils import Framework, Source, Task

variants = ["microsoft/speecht5_asr"]


@pytest.mark.nightly
@pytest.mark.xfail()
@pytest.mark.parametrize("variant", variants)
def test_speecht5_text_to_speech(variant, forge_property_recorder):
    # Record Forge properties
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PADDLE,
        model="speecht5",
        variant=variant,
        source=Source.PADDLENLP,
        task=Task.TEXT_TO_SPEECH,
    )
    forge_property_recorder.record_group("generality")

    # Load Model and Tokenizer
    model = SpeechT5ForTextToSpeech.from_pretrained(variant)
    processor = SpeechT5Processor.from_pretrained(variant)

    # Load sample
    inputs = processor(text="Hi, nice to meet you!", return_tensors="pd")
    speaker_embeddings = paddle.zeros((1, 512))
    decoder_input_values = paddle.zeros((1, 1, 80))

    # Test framework model
    outputs = model(
        input_ids=inputs["input_ids"], speaker_embeddings=speaker_embeddings, decoder_input_values=decoder_input_values
    )

    class WrappedSpeechT5Model(paddle.nn.Layer):
        def __init__(self, model):
            super(WrappedSpeechT5Model, self).__init__()
            self.model = model

        def forward(self, input_ids, speaker_embeddings, decoder_input_values):
            return self.model(
                input_ids=input_ids, speaker_embeddings=speaker_embeddings, decoder_input_values=decoder_input_values
            )

    model = WrappedSpeechT5Model(model)

    inputs = [inputs["input_ids"], speaker_embeddings, decoder_input_values]
    framework_model, _ = paddle_trace(model, inputs=inputs)

    # Compile Model
    compiled_model = forge.compile(
        framework_model, inputs, forge_property_handler=forge_property_recorder, module_name=module_name
    )

    # Verify
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
