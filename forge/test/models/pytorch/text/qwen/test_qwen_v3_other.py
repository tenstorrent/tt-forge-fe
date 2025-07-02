# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    ModelPriority,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.models_utils import get_detailed_instruct
from test.utils import download_model
import forge 
from transformers import AutoModel, AutoTokenizer
from loguru import logger

variants = [
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/QwQ-32B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
]


@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
@pytest.mark.xfail
def test_qwen3(variant):

    # Record Forge Property
    record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.QWENV3,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    pytest.xfail(reason="Requires upgrade of `transformers` version")

variants = ["Qwen/Qwen3-Embedding-0.6B"]
@pytest.mark.parametrize("variant", variants)
def test_qwen3_embedding(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.QWENV3,
        variant=variant,
        task=Task.SENTENCE_EMBEDDING_GENERATION,
        source=Source.HUGGINGFACE,
    )

    # Load model and tokenizer
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    framework_model = download_model(AutoModel.from_pretrained, variant, use_cache=False)
    framework_model.eval()

    logger.info("framework_model={}",framework_model)

    has_trainable_params = False
    for name, param in framework_model.named_parameters():
        if param.requires_grad:
            has_trainable_params = True
            logger.info(f"Trainable parameter found: {name}")
            param.requires_grad = False  # Freeze it

    if has_trainable_params:
        logger.info("Trainable parameters found and frozen.")
    else:
        logger.info("No trainable parameters found.")

    # prepare input
    task = "Given a web search query, retrieve relevant passages that answer the query"

    queries = [
        get_detailed_instruct(task, "What is the capital of China?"),
        get_detailed_instruct(task, "Explain gravity"),
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]
    input_texts = queries + documents

    # Tokenize the input texts
    input_tokens = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    inputs = [input_tokens["input_ids"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
