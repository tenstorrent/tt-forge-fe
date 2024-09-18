# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import Union

from ..tensor import Tensor
from ..parameter import Parameter
from .common import ForgeOp as op

def Embedding(
        name: str, 
        indices: Tensor,
        embedding_table: Union[Tensor, Parameter]) -> Tensor:
    """
    Embedding lookup

    Parameters
    ----------
    name: str
        Op name, unique to the module, or leave blank to autoset

    indices: Tensor
        Integer tensor, the elements of which are used to index into the embedding table

    embedding_table: Tensor
        Dictionary of embeddings
    """

    return op("embedding", name, indices, embedding_table).get_tensor()
