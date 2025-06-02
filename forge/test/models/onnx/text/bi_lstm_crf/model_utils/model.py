# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from bi_lstm_crf import BiRnnCrf


class BiRnnCrfWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, xs):
        scores, tag_seq = self.model(xs)

        if isinstance(tag_seq, list):
            tag_seq = torch.tensor(tag_seq, dtype=torch.long)
        return scores, tag_seq


def get_model(test_sentence):
    word_to_ix = {"apple": 0, "corporation": 1, "is": 2, "in": 3, "georgia": 4, "<PAD>": 5}
    tag_to_ix = {"B": 0, "I": 1, "O": 2}

    embedding_dim = len(word_to_ix) - 1
    hidden_dim = len(word_to_ix) - 2
    num_rnn_layers = 1
    model = BiRnnCrf(
        vocab_size=len(word_to_ix),
        tagset_size=len(tag_to_ix),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_rnn_layers=num_rnn_layers,
        rnn="lstm",
    )

    for word in test_sentence:
        assert word in word_to_ix, f"Error: '{word}' is not in dictionary!"

    test_input = torch.tensor([[word_to_ix[w] for w in test_sentence]], dtype=torch.long)

    return BiRnnCrfWrapper(model), test_input
