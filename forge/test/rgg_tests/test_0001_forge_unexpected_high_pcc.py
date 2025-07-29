# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import forge
from test.operators.utils import TensorUtils

import pytest
from .verify import verify_module

from forge import ForgeModule, Tensor


class GeneratedTestModel_0_885440(ForgeModule):
    # graph_builder: RandomGraphAlgorithm
    # id: healthy_forge_random_graph_algorithm_default_0_885440
    # params.test_index: 0
    # params.random_seed: 885440

    def __init__(
        self, module_name: str = "Buda Test GeneratedTestModel_healthy_forge_random_graph_algorithm_default_0_885440"
    ):
        super(GeneratedTestModel_0_885440, self).__init__(module_name)
        self.testname = "Operator Test GeneratedTestModel_healthy_forge_random_graph_algorithm_default_0_885440"

        self.add_constant("nconst1")
        self.set_constant(
            "nconst1",
            TensorUtils.create_torch_constant(input_shape=(8, 7, 18, 61), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst2")
        self.set_constant(
            "nconst2",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 18, 61), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst3")
        self.set_constant(
            "nconst3",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 18, 6), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst4")
        self.set_constant(
            "nconst4",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 18, 61), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst5")
        self.set_constant(
            "nconst5",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 18, 6), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst6")
        self.set_constant(
            "nconst6",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 18, 6), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst7")
        self.set_constant(
            "nconst7",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 18, 1), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst8")
        self.set_constant(
            "nconst8",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 36, 61), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst9")
        self.set_constant(
            "nconst9",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 18, 36), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst10")
        self.set_constant(
            "nconst10",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 18, 61), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst11")
        self.set_constant(
            "nconst11",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 61, 3), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst12")
        self.set_constant(
            "nconst12",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 18, 1), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst13")
        self.set_constant(
            "nconst13",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 18, 1), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst14")
        self.set_constant(
            "nconst14",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 18, 1), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst15")
        self.set_constant(
            "nconst15",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 18, 1), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst16")
        self.set_constant(
            "nconst16",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 18, 1), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst17")
        self.set_constant(
            "nconst17",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 18, 1), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst18")
        self.set_constant(
            "nconst18",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 18, 2), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst19")
        self.set_constant(
            "nconst19",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 18, 1), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst20")
        self.set_constant(
            "nconst20",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 18, 1), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst21")
        self.set_constant(
            "nconst21",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 18, 1), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst22")
        self.set_constant(
            "nconst22",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 18, 26), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst23")
        self.set_constant(
            "nconst23",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 18, 26), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst24")
        self.set_constant(
            "nconst24",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 18, 26), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("iconst1")
        self.set_constant(
            "iconst1",
            TensorUtils.create_torch_constant(input_shape=(8, 9, 21, 61), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("iconst2")
        self.set_constant(
            "iconst2",
            TensorUtils.create_torch_constant(input_shape=(8, 9, 18, 21), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("iconst3")
        self.set_constant(
            "iconst3",
            TensorUtils.create_torch_constant(input_shape=(8, 7, 9, 61), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("iconst4")
        self.set_constant(
            "iconst4",
            TensorUtils.create_torch_constant(input_shape=(8, 7, 7, 61), value_range=(-1, 1), random_seed=885440),
        )

    def forward(
        self,
        in_value1: forge.Tensor,
        in_value2: forge.Tensor,
        in_value3: forge.Tensor,
        in_value4: forge.Tensor,
        in_value5: forge.Tensor,
        in_value6: forge.Tensor,
        in_value7: forge.Tensor,
    ) -> forge.Tensor:

        # shapes: [(8, 9, 21, 61), (8, 9, 21, 61)] -> (8, 9, 21, 61)
        inputs = [in_value1, self.get_constant("iconst1")]
        v = forge.op.Equal(
            "op1",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 9, 18, 21), (8, 9, 21, 61)] -> (8, 9, 18, 61)
        inputs = [self.get_constant("iconst2"), v]
        op2 = forge.op.Matmul(
            "op2",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 61)] -> (8, 53, 18, 61)
        inputs = [in_value2]
        v = forge.op.Reciprocal(
            "op3",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 61)] -> (8, 53, 18, 61)
        inputs = [v]
        op4 = forge.op.Reciprocal(
            "op4",
            inputs[0],
        )

        # shapes: [(8, 7, 18, 61)] -> (8, 7, 18, 61)
        inputs = [self.get_constant("nconst1")]
        op5 = forge.op.Cosine(
            "op5",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 61), (8, 53, 18, 61)] -> (8, 53, 18, 61)
        inputs = [self.get_constant("nconst2"), op4]
        op6 = forge.op.Subtract(
            "op6",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 7, 18, 61), (8, 7, 18, 61)] -> (8, 7, 18, 61)
        inputs = [op5, in_value3]
        op7 = forge.op.Min(
            "op7",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 6)] -> (8, 53, 18, 6)
        inputs = [self.get_constant("nconst3")]
        op8 = forge.op.Sine(
            "op8",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 61), (8, 53, 18, 61)] -> (8, 53, 18, 61)
        inputs = [op6, op4]
        op9 = forge.op.NotEqual(
            "op9",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 7, 9, 61), (8, 7, 1, 61), (8, 7, 7, 61), (8, 7, 1, 61)] -> (8, 7, 18, 61)
        inputs = [self.get_constant("iconst3"), in_value4, self.get_constant("iconst4"), in_value4]
        op10 = forge.op.Concatenate("op10", inputs[0], inputs[1], inputs[2], inputs[3], axis=2)

        # shapes: [(8, 11, 18, 61)] -> (8, 11, 18, 61)
        inputs = [in_value5]
        op11 = forge.op.Relu(
            "op11",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 61)] -> (8, 53, 18, 61)
        inputs = [self.get_constant("nconst4")]
        op12 = forge.op.Tanh(
            "op12",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 6)] -> (8, 53, 18, 6)
        inputs = [op8]
        op13 = forge.op.Tanh(
            "op13",
            inputs[0],
        )

        # shapes: [(8, 7, 18, 61), (8, 7, 18, 61)] -> (8, 7, 18, 61)
        inputs = [op10, op5]
        op14 = forge.op.Equal(
            "op14",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 7, 18, 61), (8, 7, 18, 61)] -> (8, 7, 18, 61)
        inputs = [op7, op5]
        op15 = forge.op.Greater(
            "op15",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 61)] -> (8, 53, 18, 61)
        inputs = [op12]
        op16 = forge.op.Tanh(
            "op16",
            inputs[0],
        )

        # shapes: [(8, 11, 18, 61)] -> (8, 11, 18, 61)
        inputs = [op11]
        op17 = forge.op.Atan(
            "op17",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 6), (8, 53, 18, 6)] -> (8, 53, 18, 6)
        inputs = [self.get_constant("nconst5"), op13]
        op18 = forge.op.Divide(
            "op18",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 61)] -> (8, 53, 18, 61)
        inputs = [op16]
        op19 = forge.op.Reciprocal(
            "op19",
            inputs[0],
        )

        # shapes: [(8, 7, 18, 61), (8, 7, 18, 61)] -> (8, 7, 18, 61)
        inputs = [op15, op14]
        v = forge.op.Max(
            "op20",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 7, 18, 61), (8, 7, 18, 61)] -> (8, 7, 18, 61)
        inputs = [v, op14]
        op21 = forge.op.Less(
            "op21",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 61), (8, 53, 18, 61)] -> (8, 53, 18, 61)
        inputs = [op19, op9]
        op22 = forge.op.Max(
            "op22",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 6), (8, 53, 18, 6)] -> (8, 53, 18, 6)
        inputs = [op13, op8]
        v = forge.op.Less(
            "op23",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 6), (8, 53, 18, 6)] -> (8, 53, 18, 6)
        inputs = [v, self.get_constant("nconst6")]
        op24 = forge.op.Multiply(
            "op24",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 61)] -> (8, 53, 18, 61)
        inputs = [op22]
        op25 = forge.op.Identity(
            "op25",
            inputs[0],
        )

        # shapes: [(8, 9, 18, 61), (8, 11, 18, 61), (8, 7, 18, 61), (8, 24, 18, 61), (8, 2, 18, 61)] -> (8, 53, 18, 61)
        inputs = [op2, op17, op21, in_value6, in_value7]
        v = forge.op.Concatenate("op26", inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], axis=1)

        # shapes: [(8, 53, 18, 61), (8, 53, 18, 61)] -> (8, 53, 18, 61)
        inputs = [v, op22]
        op27 = forge.op.Less(
            "op27",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 6), (8, 53, 18, 6)] -> (8, 53, 18, 6)
        inputs = [op24, op18]
        v = forge.op.Multiply(
            "op28",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 6)] -> (8, 53, 18, 6)
        inputs = [v]
        op29 = forge.op.Relu(
            "op29",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 61), (8, 53, 18, 61)] -> (8, 53, 18, 61)
        inputs = [op27, op25]
        op30 = forge.op.Divide(
            "op30",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 1)] -> (8, 53, 18, 1)
        inputs = [self.get_constant("nconst7")]
        op31 = forge.op.Sigmoid(
            "op31",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 6)] -> (8, 53, 18, 6)
        inputs = [op29]
        op32 = forge.op.Identity(
            "op32",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 6)] -> (8, 53, 18, 6)
        inputs = [op32]
        v = forge.op.Cosine(
            "op33",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 6)] -> (8, 53, 18, 6)
        inputs = [v]
        op34 = forge.op.Relu(
            "op34",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 61)] -> (8, 53, 18, 61)
        inputs = [op30]
        op35 = forge.op.CumSum("op35", inputs[0], dim=-3)

        # shapes: [(8, 53, 18, 6), (8, 53, 18, 6)] -> (8, 53, 18, 6)
        inputs = [op34, op32]
        op36 = forge.op.Add(
            "op36",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 1)] -> (8, 53, 18, 1)
        inputs = [op31]
        op37 = forge.op.Abs(
            "op37",
            inputs[0],
        )

        # shapes: [(8, 53, 36, 61)] -> (8, 53, 36, 61)
        inputs = [self.get_constant("nconst8")]
        v = forge.op.Atan(
            "op38",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 36), (8, 53, 36, 61)] -> (8, 53, 18, 61)
        inputs = [self.get_constant("nconst9"), v]
        op39 = forge.op.Matmul(
            "op39",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 1)] -> (8, 53, 18, 1)
        inputs = [op37]
        op40 = forge.op.LeakyRelu("op40", inputs[0], alpha=55.56383235223071)

        # shapes: [(8, 53, 18, 61)] -> (8, 53, 18, 61)
        inputs = [op39]
        op41 = forge.op.Cosine(
            "op41",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 61)] -> (8, 53, 18, 61)
        inputs = [op35]
        op42 = forge.op.Reciprocal(
            "op42",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 1)] -> (8, 53, 18, 1)
        inputs = [op40]
        op43 = forge.op.LeakyRelu("op43", inputs[0], alpha=29.06277359174486)

        # shapes: [(8, 53, 18, 61)] -> (8, 53, 18, 61)
        inputs = [op42]
        op44 = forge.op.Erf(
            "op44",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 61), (8, 53, 18, 61)] -> (8, 53, 18, 61)
        inputs = [op42, op41]
        op45 = forge.op.Divide(
            "op45",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 1), (8, 53, 18, 1)] -> (8, 53, 18, 1)
        inputs = [op43, op37]
        op46 = forge.op.Less(
            "op46",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 61)] -> (8, 53, 18, 61)
        inputs = [op45]
        op47 = forge.op.Clip("op47", inputs[0], min=33.717802047372736, max=28.673860121064575)

        # shapes: [(8, 53, 18, 1), (8, 53, 18, 1)] -> (8, 53, 18, 1)
        inputs = [op46, op43]
        op48 = forge.op.Multiply(
            "op48",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 6)] -> (8, 53, 18, 6)
        inputs = [op36]
        op49 = forge.op.Clip("op49", inputs[0], min=54.01173248743295, max=52.736990080791756)

        # shapes: [(8, 53, 18, 61), (8, 53, 18, 61)] -> (8, 53, 18, 61)
        inputs = [op47, op44]
        v = forge.op.Equal(
            "op50",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 61), (8, 53, 18, 61)] -> (8, 53, 18, 61)
        inputs = [self.get_constant("nconst10"), v]
        v = forge.op.Max(
            "op51",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 61), (8, 53, 61, 3)] -> (8, 53, 18, 3)
        inputs = [v, self.get_constant("nconst11")]
        op52 = forge.op.Matmul(
            "op52",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 6)] -> (8, 53, 18, 6)
        inputs = [op49]
        op53 = forge.op.Erf(
            "op53",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 1), (8, 53, 18, 1)] -> (8, 53, 18, 1)
        inputs = [op46, op43]
        op54 = forge.op.Multiply(
            "op54",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 1), (8, 53, 18, 1)] -> (8, 53, 18, 1)
        inputs = [op48, self.get_constant("nconst12")]
        op55 = forge.op.Max(
            "op55",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 3), (8, 53, 18, 6), (8, 53, 18, 1), (8, 53, 18, 1), (8, 53, 18, 1), (8, 53, 18, 1), (8, 53, 18, 1), (8, 53, 18, 1), (8, 53, 18, 1)] -> (8, 53, 18, 16)
        inputs = [
            op52,
            op53,
            self.get_constant("nconst15"),
            op55,
            op54,
            op43,
            self.get_constant("nconst14"),
            op37,
            self.get_constant("nconst13"),
        ]
        op56 = forge.op.Concatenate(
            "op56",
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[5],
            inputs[6],
            inputs[7],
            inputs[8],
            axis=3,
        )

        # shapes: [(8, 53, 18, 1), (8, 53, 18, 1)] -> (8, 53, 18, 1)
        inputs = [self.get_constant("nconst17"), self.get_constant("nconst16")]
        op57 = forge.op.Min(
            "op57",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 16)] -> (8, 53, 18, 16)
        inputs = [op56]
        v = forge.op.Cosine(
            "op58",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 16)] -> (8, 53, 18, 16)
        inputs = [v]
        op59 = forge.op.Abs(
            "op59",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 1), (8, 53, 18, 1)] -> (8, 53, 18, 1)
        inputs = [op57, op55]
        op60 = forge.op.Greater(
            "op60",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 2)] -> (8, 53, 18, 2)
        inputs = [self.get_constant("nconst18")]
        op61 = forge.op.Relu(
            "op61",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 1)] -> (8, 53, 18, 1)
        inputs = [op60]
        v = forge.op.Sigmoid(
            "op62",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 1)] -> (8, 53, 18, 1)
        inputs = [v]
        op63 = forge.op.Concatenate("op63", inputs[0], axis=3)

        # shapes: [(8, 53, 18, 1)] -> (8, 53, 18, 1)
        inputs = [op63]
        op64 = forge.op.Clip("op64", inputs[0], min=94.79877505075822, max=16.696794681688264)

        # shapes: [(8, 53, 18, 1), (8, 53, 18, 1)] -> (8, 53, 18, 1)
        inputs = [op64, op63]
        v = forge.op.Max(
            "op65",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 1), (8, 53, 18, 1)] -> (8, 53, 18, 1)
        inputs = [v, op64]
        op66 = forge.op.Min(
            "op66",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 1)] -> (8, 53, 18, 1)
        inputs = [op66]
        op67 = forge.op.Abs(
            "op67",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 1), (8, 53, 18, 1)] -> (8, 53, 18, 1)
        inputs = [op67, self.get_constant("nconst19")]
        v = forge.op.Greater(
            "op68",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 1)] -> (8, 53, 18, 1)
        inputs = [v]
        op69 = forge.op.Cosine(
            "op69",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 1), (8, 53, 18, 1)] -> (8, 53, 18, 1)
        inputs = [op69, op66]
        op70 = forge.op.Less(
            "op70",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 1), (8, 53, 18, 1)] -> (8, 53, 18, 1)
        inputs = [op70, op69]
        v = forge.op.Max(
            "op71",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 1), (8, 53, 18, 1)] -> (8, 53, 18, 1)
        inputs = [v, op70]
        v = forge.op.Subtract(
            "op72",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 16), (8, 53, 18, 2), (8, 53, 18, 1), (8, 53, 18, 1), (8, 53, 18, 1), (8, 53, 18, 1), (8, 53, 18, 1), (8, 53, 18, 1), (8, 53, 18, 1), (8, 53, 18, 1)] -> (8, 53, 18, 26)
        inputs = [
            op59,
            op61,
            self.get_constant("nconst21"),
            v,
            self.get_constant("nconst20"),
            op70,
            op69,
            op67,
            v,
            op66,
        ]
        v = forge.op.Concatenate(
            "op73",
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[5],
            inputs[6],
            inputs[7],
            inputs[8],
            inputs[9],
            axis=3,
        )

        # shapes: [(8, 53, 18, 26), (8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [self.get_constant("nconst22"), v]
        v = forge.op.NotEqual(
            "op74",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [v]
        v = forge.op.Cosine(
            "op75",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 26), (8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [v, v]
        v = forge.op.Less(
            "op76",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 26), (8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [self.get_constant("nconst23"), v]
        v = forge.op.NotEqual(
            "op77",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [v]
        v = forge.op.Exp(
            "op78",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [v]
        v = forge.op.Sigmoid(
            "op79",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [v]
        op80 = forge.op.Cosine(
            "op80",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [self.get_constant("nconst24")]
        op81 = forge.op.Sigmoid(
            "op81",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 26), (8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [op81, op80]
        v = forge.op.Max(
            "op82",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [v]
        v = forge.op.Sigmoid(
            "op83",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [v]
        v = forge.op.Abs(
            "op84",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 26), (8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [v, op81]
        op85 = forge.op.Add(
            "op85",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [op85]
        v = forge.op.Tanh(
            "op86",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 26), (8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [v, op85]
        v = forge.op.Multiply(
            "op87",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [v]
        v = forge.op.Gelu(
            "op88",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [v]
        v = forge.op.Gelu(
            "op89",
            inputs[0],
        )

        return v


# @pytest.mark.xfail(reason="The model triggers a bug.")
def test_gen_model_0_885440(test_device):

    input_shapes = [
        (8, 9, 21, 61),
        (8, 53, 18, 61),
        (8, 7, 18, 61),
        (8, 7, 1, 61),
        (8, 11, 18, 61),
        (8, 24, 18, 61),
        (8, 2, 18, 61),
    ]
    model = GeneratedTestModel_0_885440("pytest_gen_model_healthy_forge_random_graph_algorithm_default_0_885440")

    verify_module(model, input_shapes, random_seed=885440)
