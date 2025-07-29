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
            TensorUtils.create_torch_constant(input_shape=(8, 53, 20, 30), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst2")
        self.set_constant(
            "nconst2",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 20, 3), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst3")
        self.set_constant(
            "nconst3",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 20, 6), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst4")
        self.set_constant(
            "nconst4",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 20, 26), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst5")
        self.set_constant(
            "nconst5",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 30, 1), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst6")
        self.set_constant(
            "nconst6",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 30, 1), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst7")
        self.set_constant(
            "nconst7",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 30, 2), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst8")
        self.set_constant(
            "nconst8",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 30, 20), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst9")
        self.set_constant(
            "nconst9",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 30, 26), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst10")
        self.set_constant(
            "nconst10",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 30, 26), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst11")
        self.set_constant(
            "nconst11",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 30, 26), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst12")
        self.set_constant(
            "nconst12",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 18, 30), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst13")
        self.set_constant(
            "nconst13",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 18, 26), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst14")
        self.set_constant(
            "nconst14",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 18, 26), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst15")
        self.set_constant(
            "nconst15",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 18, 26), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst16")
        self.set_constant(
            "nconst16",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 18, 26), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst17")
        self.set_constant(
            "nconst17",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 18, 26), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("nconst18")
        self.set_constant(
            "nconst18",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 18, 26), value_range=(-1, 1), random_seed=885440),
        )
        self.add_constant("iconst1")
        self.set_constant(
            "iconst1",
            TensorUtils.create_torch_constant(input_shape=(8, 53, 36, 3), value_range=(-1, 1), random_seed=885440),
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
        in_value8: forge.Tensor,
        in_value9: forge.Tensor,
        in_value10: forge.Tensor,
    ) -> forge.Tensor:

        # shapes: [(8, 53, 20, 1)] -> (8, 53, 20, 1)
        inputs = [in_value1]
        op1 = forge.op.LeakyRelu("op1", inputs[0], alpha=47.34085647160601)

        # shapes: [(8, 53, 20, 6), (8, 53, 20, 6)] -> (8, 53, 20, 6)
        inputs = [in_value2, in_value2]
        op2 = forge.op.Greater(
            "op2",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 20, 30)] -> (8, 53, 20, 30)
        inputs = [in_value3]
        op3 = forge.op.Reciprocal(
            "op3",
            inputs[0],
        )

        # shapes: [(8, 53, 20, 6)] -> (8, 53, 20, 6)
        inputs = [op2]
        op4 = forge.op.Sine(
            "op4",
            inputs[0],
        )

        # shapes: [(8, 53, 20, 30), (8, 53, 20, 30)] -> (8, 53, 20, 30)
        inputs = [self.get_constant("nconst1"), op3]
        op5 = forge.op.Add(
            "op5",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 20, 36), (8, 53, 36, 3)] -> (8, 53, 20, 3)
        inputs = [in_value4, self.get_constant("iconst1")]
        op6 = forge.op.Matmul(
            "op6",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 20, 4)] -> (8, 53, 20, 4)
        inputs = [in_value5]
        op7 = forge.op.Atan(
            "op7",
            inputs[0],
        )

        # shapes: [(8, 53, 20, 30)] -> (8, 53, 20, 30)
        inputs = [op5]
        op8 = forge.op.Relu(
            "op8",
            inputs[0],
        )

        # shapes: [(8, 53, 20, 6), (8, 53, 20, 6)] -> (8, 53, 20, 6)
        inputs = [in_value2, in_value6]
        op9 = forge.op.Min(
            "op9",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 20, 3)] -> (8, 53, 20, 3)
        inputs = [self.get_constant("nconst2")]
        op10 = forge.op.LogicalNot(
            "op10",
            inputs[0],
        )

        # shapes: [(8, 53, 20, 6), (8, 53, 20, 6)] -> (8, 53, 20, 6)
        inputs = [op9, self.get_constant("nconst3")]
        op11 = forge.op.Less(
            "op11",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 20, 6), (8, 53, 20, 6)] -> (8, 53, 20, 6)
        inputs = [op9, op4]
        op12 = forge.op.Divide(
            "op12",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 20, 30), (8, 53, 30, 1)] -> (8, 53, 20, 1)
        inputs = [op8, in_value7]
        op13 = forge.op.Matmul(
            "op13",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 20, 6), (8, 53, 20, 6)] -> (8, 53, 20, 6)
        inputs = [op12, op11]
        v = forge.op.Greater(
            "op14",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 20, 6), (8, 53, 20, 3), (8, 53, 20, 6), (8, 53, 20, 3), (8, 53, 20, 4), (8, 53, 20, 1), (8, 53, 20, 1), (8, 53, 20, 1), (8, 53, 20, 1)] -> (8, 53, 20, 26)
        inputs = [v, op10, op12, op6, op7, op13, op1, in_value1, in_value8]
        v = forge.op.Concatenate(
            "op15",
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[5],
            inputs[6],
            inputs[7],
            inputs[8],
            axis=-1,
        )

        # shapes: [(8, 53, 20, 26), (8, 53, 20, 26)] -> (8, 53, 20, 26)
        inputs = [self.get_constant("nconst4"), v]
        v = forge.op.Max(
            "op16",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 20, 26)] -> (8, 53, 20, 26)
        inputs = [v]
        v = forge.op.Abs(
            "op17",
            inputs[0],
        )

        # shapes: [(8, 53, 20, 26)] -> (8, 53, 20, 26)
        inputs = [v]
        op18 = forge.op.Tanh(
            "op18",
            inputs[0],
        )

        # shapes: [(8, 53, 30, 11), (8, 53, 30, 2), (8, 53, 30, 2), (8, 53, 30, 1), (8, 53, 30, 1), (8, 53, 30, 1), (8, 53, 30, 2)] -> (8, 53, 30, 20)
        inputs = [
            in_value9,
            in_value10,
            self.get_constant("nconst7"),
            in_value7,
            self.get_constant("nconst6"),
            self.get_constant("nconst5"),
            in_value10,
        ]
        op19 = forge.op.Concatenate(
            "op19", inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], axis=3
        )

        # shapes: [(8, 53, 20, 26)] -> (8, 53, 20, 26)
        inputs = [op18]
        op20 = forge.op.Reciprocal(
            "op20",
            inputs[0],
        )

        # shapes: [(8, 53, 20, 26)] -> (8, 53, 20, 26)
        inputs = [op18]
        op21 = forge.op.Sine(
            "op21",
            inputs[0],
        )

        # shapes: [(8, 53, 30, 20)] -> (8, 53, 30, 20)
        inputs = [op19]
        op22 = forge.op.Reciprocal(
            "op22",
            inputs[0],
        )

        # shapes: [(8, 53, 20, 26)] -> (8, 53, 20, 26)
        inputs = [op21]
        v = forge.op.LeakyRelu("op23", inputs[0], alpha=0.12770076002756037)

        # shapes: [(8, 53, 20, 26), (8, 53, 20, 26)] -> (8, 53, 20, 26)
        inputs = [v, op20]
        v = forge.op.GreaterEqual(
            "op24",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 20, 26), (8, 53, 20, 26)] -> (8, 53, 20, 26)
        inputs = [v, op20]
        op25 = forge.op.Subtract(
            "op25",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 30, 20)] -> (8, 53, 30, 20)
        inputs = [op22]
        v = forge.op.Identity(
            "op26",
            inputs[0],
        )

        # shapes: [(8, 53, 30, 20), (8, 53, 30, 20)] -> (8, 53, 30, 20)
        inputs = [self.get_constant("nconst8"), v]
        op27 = forge.op.Equal(
            "op27",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 20, 26)] -> (8, 53, 20, 26)
        inputs = [op25]
        v = forge.op.Identity(
            "op28",
            inputs[0],
        )

        # shapes: [(8, 53, 20, 26)] -> (8, 53, 20, 26)
        inputs = [v]
        op29 = forge.op.LeakyRelu("op29", inputs[0], alpha=38.438232742275105)

        # shapes: [(8, 53, 20, 26)] -> (8, 53, 20, 26)
        inputs = [op29]
        v = forge.op.Gelu(
            "op30",
            inputs[0],
        )

        # shapes: [(8, 53, 20, 26), (8, 53, 20, 26)] -> (8, 53, 20, 26)
        inputs = [v, op29]
        v = forge.op.NotEqual(
            "op31",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 30, 20), (8, 53, 20, 26)] -> (8, 53, 30, 26)
        inputs = [op27, v]
        op32 = forge.op.Matmul(
            "op32",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 30, 26), (8, 53, 30, 26)] -> (8, 53, 30, 26)
        inputs = [op32, self.get_constant("nconst9")]
        v = forge.op.Subtract(
            "op33",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 30, 26)] -> (8, 53, 30, 26)
        inputs = [v]
        v = forge.op.Clip("op34", inputs[0], min=55.56383235223071, max=9.5451981236637)

        # shapes: [(8, 53, 30, 26), (8, 53, 30, 26)] -> (8, 53, 30, 26)
        inputs = [v, op32]
        v = forge.op.Multiply(
            "op35",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 30, 26), (8, 53, 30, 26)] -> (8, 53, 30, 26)
        inputs = [v, v]
        v = forge.op.NotEqual(
            "op36",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 30, 26)] -> (8, 53, 30, 26)
        inputs = [v]
        v = forge.op.LeakyRelu("op37", inputs[0], alpha=29.06277359174486)

        # shapes: [(8, 53, 30, 26), (8, 53, 30, 26)] -> (8, 53, 30, 26)
        inputs = [self.get_constant("nconst10"), v]
        op38 = forge.op.Min(
            "op38",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 30, 26)] -> (8, 53, 30, 26)
        inputs = [op38]
        v = forge.op.Clip("op39", inputs[0], min=33.717802047372736, max=28.673860121064575)

        # shapes: [(8, 53, 30, 26)] -> (8, 53, 30, 26)
        inputs = [v]
        v = forge.op.Clip("op40", inputs[0], min=54.01173248743295, max=52.736990080791756)

        # shapes: [(8, 53, 30, 26)] -> (8, 53, 30, 26)
        inputs = [v]
        op41 = forge.op.CumSum("op41", inputs[0], dim=1)

        # shapes: [(8, 53, 30, 26)] -> (8, 53, 30, 26)
        inputs = [op41]
        op42 = forge.op.Sine(
            "op42",
            inputs[0],
        )

        # shapes: [(8, 53, 30, 26), (8, 53, 30, 26)] -> (8, 53, 30, 26)
        inputs = [op41, op38]
        v = forge.op.Multiply(
            "op43",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 30, 26)] -> (8, 53, 30, 26)
        inputs = [v]
        op44 = forge.op.LeakyRelu("op44", inputs[0], alpha=47.058683034561064)

        # shapes: [(8, 53, 30, 26)] -> (8, 53, 30, 26)
        inputs = [op42]
        op45 = forge.op.LogicalNot(
            "op45",
            inputs[0],
        )

        # shapes: [(8, 53, 30, 26)] -> (8, 53, 30, 26)
        inputs = [op44]
        v = forge.op.Sigmoid(
            "op46",
            inputs[0],
        )

        # shapes: [(8, 53, 30, 26)] -> (8, 53, 30, 26)
        inputs = [v]
        v = forge.op.Sigmoid(
            "op47",
            inputs[0],
        )

        # shapes: [(8, 53, 30, 26), (8, 53, 30, 26)] -> (8, 53, 30, 26)
        inputs = [v, op45]
        op48 = forge.op.Equal(
            "op48",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 30, 26)] -> (8, 53, 30, 26)
        inputs = [op48]
        op49 = forge.op.Abs(
            "op49",
            inputs[0],
        )

        # shapes: [(8, 53, 30, 26)] -> (8, 53, 30, 26)
        inputs = [self.get_constant("nconst11")]
        v = forge.op.Abs(
            "op50",
            inputs[0],
        )

        # shapes: [(8, 53, 30, 26)] -> (8, 53, 30, 26)
        inputs = [v]
        op51 = forge.op.Exp(
            "op51",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 30)] -> (8, 53, 18, 30)
        inputs = [self.get_constant("nconst12")]
        op52 = forge.op.Gelu(
            "op52",
            inputs[0],
        )

        # shapes: [(8, 53, 30, 26), (8, 53, 30, 26)] -> (8, 53, 30, 26)
        inputs = [op51, op49]
        op53 = forge.op.Subtract(
            "op53",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 30, 26), (8, 53, 30, 26)] -> (8, 53, 30, 26)
        inputs = [op53, op48]
        op54 = forge.op.Multiply(
            "op54",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 30, 26), (8, 53, 30, 26)] -> (8, 53, 30, 26)
        inputs = [op54, op53]
        v = forge.op.NotEqual(
            "op55",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 30, 26), (8, 53, 30, 26)] -> (8, 53, 30, 26)
        inputs = [v, op54]
        v = forge.op.Max(
            "op56",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 30), (8, 53, 30, 26)] -> (8, 53, 18, 26)
        inputs = [op52, v]
        op57 = forge.op.Matmul(
            "op57",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [self.get_constant("nconst13")]
        op58 = forge.op.Abs(
            "op58",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 26), (8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [op58, op57]
        op59 = forge.op.Min(
            "op59",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [op58]
        v = forge.op.Relu(
            "op60",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [v]
        v = forge.op.Sigmoid(
            "op61",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 26), (8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [v, op59]
        v = forge.op.Greater(
            "op62",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [v]
        v = forge.op.Erf(
            "op63",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 26), (8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [v, self.get_constant("nconst14")]
        op64 = forge.op.Min(
            "op64",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 26), (8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [op64, self.get_constant("nconst15")]
        op65 = forge.op.Add(
            "op65",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [op65]
        op66 = forge.op.Abs(
            "op66",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 26), (8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [op65, op64]
        v = forge.op.Min(
            "op67",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [v]
        op68 = forge.op.Cosine(
            "op68",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 26), (8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [op68, op66]
        op69 = forge.op.GreaterEqual(
            "op69",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 26), (8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [op69, op68]
        v = forge.op.Multiply(
            "op70",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [v]
        op71 = forge.op.Clip("op71", inputs[0], min=86.58621280690244, max=94.79877505075822)

        # shapes: [(8, 53, 18, 26), (8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [op69, op68]
        v = forge.op.Greater(
            "op72",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 26), (8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [v, op71]
        v = forge.op.Less(
            "op73",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 26), (8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [v, op71]
        v = forge.op.Less(
            "op74",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 26), (8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [self.get_constant("nconst16"), v]
        v = forge.op.Max(
            "op75",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 26), (8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [v, v]
        v = forge.op.GreaterEqual(
            "op76",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 26), (8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [self.get_constant("nconst17"), v]
        v = forge.op.NotEqual(
            "op77",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [v]
        op78 = forge.op.LeakyRelu("op78", inputs[0], alpha=66.98492183195073)

        # shapes: [(8, 53, 18, 26), (8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [op78, self.get_constant("nconst18")]
        v = forge.op.Multiply(
            "op79",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 26), (8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [v, op78]
        op80 = forge.op.Equal(
            "op80",
            inputs[0],
            inputs[1],
        )

        # shapes: [(8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [op80]
        v = forge.op.Sigmoid(
            "op81",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 26), (8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [v, op80]
        v = forge.op.Multiply(
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

        # shapes: [(8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [v]
        op85 = forge.op.Erf(
            "op85",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [op85]
        v = forge.op.Tanh(
            "op86",
            inputs[0],
        )

        # shapes: [(8, 53, 18, 26), (8, 53, 18, 26)] -> (8, 53, 18, 26)
        inputs = [v, op85]
        v = forge.op.Subtract(
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
        (8, 53, 20, 1),
        (8, 53, 20, 6),
        (8, 53, 20, 30),
        (8, 53, 20, 36),
        (8, 53, 20, 4),
        (8, 53, 20, 6),
        (8, 53, 30, 1),
        (8, 53, 20, 1),
        (8, 53, 30, 11),
        (8, 53, 30, 2),
    ]
    model = GeneratedTestModel_0_885440("pytest_gen_model_healthy_forge_random_graph_algorithm_default_0_885440")

    verify_module(model, input_shapes, random_seed=885440)
