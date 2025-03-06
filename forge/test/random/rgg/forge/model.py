# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Building Forge models


from typing import Type
from loguru import logger

from forge import ForgeModule

from .. import RandomizerGraph, ModelBuilder, StrUtils


class ForgeModelBuilder(ModelBuilder):
    def build_model(self, graph: RandomizerGraph, GeneratedTestModel: Type[ForgeModule]) -> ForgeModule:
        module_name = f"gen_model_pytest_{StrUtils.test_id(graph)}"
        forge_model = GeneratedTestModel(module_name)
        return forge_model
