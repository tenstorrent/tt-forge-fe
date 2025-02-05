# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import AutoTokenizer, ModernBertModel

variants = [
    "answerdotai/ModernBERT-base",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_modernbert_test_generation(variant):
    tokenizer = AutoTokenizer.from_pretrained(variant)
    framework_model = ModernBertModel.from_pretrained(variant)  # , return_dict=False)
    # model = AutoModelForMaskedLM.from_pretrained("answerdotai/ModernBERT-base")

    inputs = tokenizer(
        "Hello, my dog is cute",
        return_tensors="pt",
        max_length=150,
        pad_to_max_length=True,
        truncation=True,
    )
    outputs = framework_model(**inputs)

    print("outputs:", outputs)
