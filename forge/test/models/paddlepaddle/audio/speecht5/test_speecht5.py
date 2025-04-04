from paddlenlp.transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5ForSpeechToText, SpeechT5ForSpeechToSpeech
import paddle
import pytest
from datasets import load_dataset

import forge
from forge.verify.verify import verify
from forge.tvm_calls.forge_utils import paddle_trace

from test.models.utils import Framework, Source, Task, build_module_name

variants = ["microsoft/speecht5_asr"]

@pytest.mark.nightly
@pytest.mark.xfail()
@pytest.mark.parametrize("variant", variants)
def test_speecht5_text_to_speech(variant, forge_property_recorder):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PADDLE,
        model="speecht5",
        variant=variant,
        source=Source.PADDLENLP,
        task=Task.TEXT_TO_SPEECH,
    )
    forge_property_recorder.record_model_name(module_name)
    forge_property_recorder.record_group("generality")

    # Load Model and Tokenizer
    model = SpeechT5ForTextToSpeech.from_pretrained(variant)
    processor = SpeechT5Processor.from_pretrained(variant)

    # Load sample
    inputs = processor(text="Hi, nice to meet you!", return_tensors="pd")
    speaker_embeddings = paddle.zeros((1, 512))  
    decoder_input_values = paddle.zeros((1,1,80))

    # Test framework model
    outputs = model(input_ids=inputs["input_ids"], speaker_embeddings=speaker_embeddings, decoder_input_values=decoder_input_values)

    class WrappedSpeechT5Model(paddle.nn.Layer):
        def __init__(self, model):
            super(WrappedSpeechT5Model, self).__init__()
            self.model = model

        def forward(self, input_ids, speaker_embeddings, decoder_input_values):
            return self.model(input_ids=input_ids, speaker_embeddings=speaker_embeddings, decoder_input_values=decoder_input_values)

    model = WrappedSpeechT5Model(model)
    
    inputs = [inputs["input_ids"], speaker_embeddings, decoder_input_values]
    framework_model,_ = paddle_trace(model, inputs=inputs)

    # Compile Model
    compiled_model = forge.compile(framework_model, inputs)

    # Verify
    verify(inputs, framework_model, compiled_model)

@pytest.mark.nightly
@pytest.mark.xfail()
@pytest.mark.parametrize("variant", variants)
def test_speecht5_speech_to_text(variant, forge_property_recorder):
    dataset = load_dataset(
    "hf-internal-testing/librispeech_asr_demo", "clean", split="validation"
    )  # doctest: +IGNORE_RESULT
    dataset = dataset.sort("id")
    sampling_rate = dataset.features["audio"].sampling_rate

    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
    model = SpeechT5ForSpeechToText.from_pretrained("microsoft/speecht5_vc", ignore_mismatched_sizes=True)
    model.eval()

    # audio file is decoded on the fly
    inputs = processor(audio=dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pd")
    decoder_input_ids = paddle.zeros_like(inputs["input_values"], dtype="int32")

    inputs = [inputs["input_values"], decoder_input_ids]
    predicted_ids = model(inputs[0], decoder_input_ids=decoder_input_ids)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    print(transcription[0])

    class WrappedSpeechT5Model(paddle.nn.Layer):
        def __init__(self, model):
            super(WrappedSpeechT5Model, self).__init__()
            self.model = model

        def forward(self, input_values, decoder_input_ids):
            return self.model(input_values=input_values, decoder_input_ids=decoder_input_ids)
        
    model = WrappedSpeechT5Model(model)

    framework_model,_ = paddle_trace(model, inputs=inputs)

    # Compile Model
    compiled_model = forge.compile(framework_model, inputs)

    # Verify
    verify(inputs, framework_model, compiled_model)

