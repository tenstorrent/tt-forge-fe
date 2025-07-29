# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import forge
from test.operators.utils import TensorUtils

import pytest
from test.random.rgg.verify import verify_module
from .verify import verify_module

from forge import ForgeModule, Tensor


class GeneratedTestModel_1_403958(ForgeModule):
    # graph_builder: RandomGraphAlgorithm
    # id: healthy_forge_random_graph_algorithm_default_1_403958
    # params.test_index: 1
    # params.random_seed: 403958

    def __init__(
        self, module_name: str = "Buda Test GeneratedTestModel_healthy_forge_random_graph_algorithm_default_1_403958"
    ):
        super(GeneratedTestModel_1_403958, self).__init__(module_name)
        self.testname = "Operator Test GeneratedTestModel_healthy_forge_random_graph_algorithm_default_1_403958"

        self.add_constant("nconst1")
        self.set_constant(
            "nconst1",
            TensorUtils.create_torch_constant(input_shape=(4, 3, 39, 63), value_range=(-1, 1), random_seed=403958),
        )
        self.add_constant("nconst2")
        self.set_constant(
            "nconst2",
            TensorUtils.create_torch_constant(input_shape=(4, 2, 1, 63), value_range=(-1, 1), random_seed=403958),
        )
        self.add_constant("nconst3")
        self.set_constant(
            "nconst3",
            TensorUtils.create_torch_constant(input_shape=(4, 6, 1, 63), value_range=(-1, 1), random_seed=403958),
        )
        self.add_constant("nconst4")
        self.set_constant(
            "nconst4",
            TensorUtils.create_torch_constant(input_shape=(4, 6, 1, 63), value_range=(-1, 1), random_seed=403958),
        )
        self.add_constant("nconst5")
        self.set_constant(
            "nconst5",
            TensorUtils.create_torch_constant(input_shape=(4, 3, 1, 39), value_range=(-1, 1), random_seed=403958),
        )
        self.add_constant("nconst6")
        self.set_constant(
            "nconst6",
            TensorUtils.create_torch_constant(input_shape=(4, 34, 4, 1), value_range=(-1, 1), random_seed=403958),
        )
        self.add_constant("nconst7")
        self.set_constant(
            "nconst7",
            TensorUtils.create_torch_constant(input_shape=(4, 34, 11, 1), value_range=(-1, 1), random_seed=403958),
        )
        self.add_constant("nconst8")
        self.set_constant(
            "nconst8",
            TensorUtils.create_torch_constant(input_shape=(4, 34, 18, 46), value_range=(-1, 1), random_seed=403958),
        )
        self.add_constant("nconst9")
        self.set_constant(
            "nconst9",
            TensorUtils.create_torch_constant(input_shape=(4, 34, 18, 1), value_range=(-1, 1), random_seed=403958),
        )
        self.add_constant("nconst10")
        self.set_constant(
            "nconst10",
            TensorUtils.create_torch_constant(input_shape=(4, 34, 18, 1), value_range=(-1, 1), random_seed=403958),
        )
        self.add_constant("nconst11")
        self.set_constant(
            "nconst11",
            TensorUtils.create_torch_constant(input_shape=(4, 17, 1, 63), value_range=(-1, 1), random_seed=403958),
        )
        self.add_constant("nconst12")
        self.set_constant(
            "nconst12",
            TensorUtils.create_torch_constant(input_shape=(4, 34, 18, 12), value_range=(-1, 1), random_seed=403958),
        )
        self.add_constant("nconst13")
        self.set_constant(
            "nconst13",
            TensorUtils.create_torch_constant(input_shape=(4, 1, 1, 63), value_range=(-1, 1), random_seed=403958),
        )
        self.add_constant("nconst14")
        self.set_constant(
            "nconst14",
            TensorUtils.create_torch_constant(input_shape=(4, 34, 9, 63), value_range=(-1, 1), random_seed=403958),
        )
        self.add_constant("nconst15")
        self.set_constant(
            "nconst15",
            TensorUtils.create_torch_constant(input_shape=(4, 34, 18, 1), value_range=(-1, 1), random_seed=403958),
        )
        self.add_constant("nconst16")
        self.set_constant(
            "nconst16",
            TensorUtils.create_torch_constant(input_shape=(4, 5, 1, 63), value_range=(-1, 1), random_seed=403958),
        )
        self.add_constant("nconst17")
        self.set_constant(
            "nconst17",
            TensorUtils.create_torch_constant(input_shape=(4, 34, 1, 63), value_range=(-1, 1), random_seed=403958),
        )
        self.add_constant("nconst18")
        self.set_constant(
            "nconst18",
            TensorUtils.create_torch_constant(input_shape=(4, 34, 1, 63), value_range=(-1, 1), random_seed=403958),
        )
        self.add_constant("nconst19")
        self.set_constant(
            "nconst19",
            TensorUtils.create_torch_constant(input_shape=(4, 34, 1, 63), value_range=(-1, 1), random_seed=403958),
        )

    def forward(
        self, in_value1: forge.Tensor, in_value2: forge.Tensor, in_value3: forge.Tensor, in_value4: forge.Tensor
    ) -> forge.Tensor:

        # shapes: [(4, 17, 1, 63), (4, 17, 1, 63)] -> (4, 17, 1, 63)
        inputs = [in_value1, in_value2]
        op1 = forge.op.Divide(
            "op1",
            inputs[0],
            inputs[1],
        )

        # shapes: [(4, 17, 1, 63)] -> (4, 17, 1, 63)
        inputs = [op1]
        op2 = forge.op.Cosine(
            "op2",
            inputs[0],
        )

        # shapes: [(4, 17, 1, 63)] -> (4, 17, 1, 63)
        inputs = [op1]
        op3 = forge.op.CumSum("op3", inputs[0], dim=1)

        # shapes: [(4, 17, 1, 63), (4, 17, 1, 63)] -> (4, 17, 1, 63)
        inputs = [op3, op2]
        op4 = forge.op.Max(
            "op4",
            inputs[0],
            inputs[1],
        )

        # shapes: [(4, 17, 1, 63)] -> (4, 17, 1, 63)
        inputs = [op3]
        v = forge.op.Relu(
            "op5",
            inputs[0],
        )

        # shapes: [(4, 17, 1, 63)] -> (4, 17, 1, 63)
        inputs = [v]
        op6 = forge.op.Concatenate("op6", inputs[0], axis=2)

        # shapes: [(4, 34, 3, 1), (4, 34, 3, 1)] -> (4, 34, 3, 1)
        inputs = [in_value3, in_value4]
        v = forge.op.Divide(
            "op7",
            inputs[0],
            inputs[1],
        )

        # shapes: [(4, 34, 3, 1)] -> (4, 34, 3, 1)
        inputs = [v]
        op8 = forge.op.Sigmoid(
            "op8",
            inputs[0],
        )

        # shapes: [(4, 17, 1, 63), (4, 17, 1, 63)] -> (4, 17, 1, 63)
        inputs = [op6, op4]
        op9 = forge.op.Greater(
            "op9",
            inputs[0],
            inputs[1],
        )

        # shapes: [(4, 34, 3, 1)] -> (4, 34, 3, 1)
        inputs = [op8]
        op10 = forge.op.Sigmoid(
            "op10",
            inputs[0],
        )

        # shapes: [(4, 17, 1, 63)] -> (4, 17, 1, 63)
        inputs = [op9]
        v = forge.op.Erf(
            "op11",
            inputs[0],
        )

        # shapes: [(4, 17, 1, 63)] -> (4, 17, 1, 63)
        inputs = [v]
        v = forge.op.Gelu(
            "op12",
            inputs[0],
        )

        # shapes: [(4, 17, 1, 63)] -> (4, 17, 1, 63)
        inputs = [v]
        op13 = forge.op.Identity(
            "op13",
            inputs[0],
        )

        # shapes: [(4, 3, 39, 63)] -> (4, 3, 39, 63)
        inputs = [self.get_constant("nconst1")]
        op14 = forge.op.Concatenate("op14", inputs[0], axis=1)

        # shapes: [(4, 2, 1, 63)] -> (4, 2, 1, 63)
        inputs = [self.get_constant("nconst2")]
        op15 = forge.op.Reciprocal(
            "op15",
            inputs[0],
        )

        # shapes: [(4, 6, 1, 63)] -> (4, 6, 1, 63)
        inputs = [self.get_constant("nconst3")]
        v = forge.op.Identity(
            "op16",
            inputs[0],
        )

        # shapes: [(4, 6, 1, 63), (4, 6, 1, 63)] -> (4, 6, 1, 63)
        inputs = [v, self.get_constant("nconst4")]
        op17 = forge.op.GreaterEqual(
            "op17",
            inputs[0],
            inputs[1],
        )

        # shapes: [(4, 3, 1, 39), (4, 3, 39, 63)] -> (4, 3, 1, 63)
        inputs = [self.get_constant("nconst5"), op14]
        op18 = forge.op.Matmul(
            "op18",
            inputs[0],
            inputs[1],
        )

        # shapes: [(4, 34, 4, 1)] -> (4, 34, 4, 1)
        inputs = [self.get_constant("nconst6")]
        op19 = forge.op.Tanh(
            "op19",
            inputs[0],
        )

        # shapes: [(4, 34, 11, 1)] -> (4, 34, 11, 1)
        inputs = [self.get_constant("nconst7")]
        v = forge.op.Sigmoid(
            "op20",
            inputs[0],
        )

        # shapes: [(4, 34, 11, 1)] -> (4, 34, 11, 1)
        inputs = [v]
        op21 = forge.op.Reciprocal(
            "op21",
            inputs[0],
        )

        # shapes: [(4, 2, 1, 63)] -> (4, 2, 1, 63)
        inputs = [op15]
        op22 = forge.op.Sine(
            "op22",
            inputs[0],
        )

        # shapes: [(4, 34, 4, 1)] -> (4, 34, 4, 1)
        inputs = [op19]
        op23 = forge.op.LeakyRelu("op23", inputs[0], alpha=80.98612765379048)

        # shapes: [(4, 2, 1, 63), (4, 2, 1, 63)] -> (4, 2, 1, 63)
        inputs = [op22, op15]
        v = forge.op.Add(
            "op24",
            inputs[0],
            inputs[1],
        )

        # shapes: [(4, 2, 1, 63)] -> (4, 2, 1, 63)
        inputs = [v]
        op25 = forge.op.Concatenate("op25", inputs[0], axis=1)

        # shapes: [(4, 17, 1, 63)] -> (4, 17, 1, 63)
        inputs = [op13]
        op26 = forge.op.Abs(
            "op26",
            inputs[0],
        )

        # shapes: [(4, 34, 11, 1)] -> (4, 34, 11, 1)
        inputs = [op21]
        v = forge.op.CumSum("op27", inputs[0], dim=-3)

        # shapes: [(4, 34, 11, 1)] -> (4, 34, 11, 1)
        inputs = [v]
        op28 = forge.op.Gelu(
            "op28",
            inputs[0],
        )

        # shapes: [(4, 2, 1, 63), (4, 2, 1, 63)] -> (4, 2, 1, 63)
        inputs = [op25, op25]
        op29 = forge.op.NotEqual(
            "op29",
            inputs[0],
            inputs[1],
        )

        # shapes: [(4, 34, 18, 46)] -> (4, 34, 18, 46)
        inputs = [self.get_constant("nconst8")]
        op30 = forge.op.Abs(
            "op30",
            inputs[0],
        )

        # shapes: [(4, 2, 1, 63), (4, 2, 1, 63)] -> (4, 2, 1, 63)
        inputs = [op29, op22]
        v = forge.op.Add(
            "op31",
            inputs[0],
            inputs[1],
        )

        # shapes: [(4, 2, 1, 63)] -> (4, 2, 1, 63)
        inputs = [v]
        op32 = forge.op.LeakyRelu("op32", inputs[0], alpha=46.54171570846957)

        # shapes: [(4, 17, 1, 63)] -> (4, 17, 1, 63)
        inputs = [op26]
        op33 = forge.op.Clip("op33", inputs[0], min=33.84217082023695, max=5.600005656558271)

        # shapes: [(4, 34, 3, 1), (4, 34, 11, 1), (4, 34, 4, 1)] -> (4, 34, 18, 1)
        inputs = [op10, op28, op23]
        op34 = forge.op.Concatenate("op34", inputs[0], inputs[1], inputs[2], axis=2)

        # shapes: [(4, 17, 1, 63), (4, 17, 1, 63)] -> (4, 17, 1, 63)
        inputs = [op33, op26]
        op35 = forge.op.Equal(
            "op35",
            inputs[0],
            inputs[1],
        )

        # shapes: [(4, 2, 1, 63)] -> (4, 2, 1, 63)
        inputs = [op32]
        v = forge.op.Relu(
            "op36",
            inputs[0],
        )

        # shapes: [(4, 2, 1, 63), (4, 2, 1, 63)] -> (4, 2, 1, 63)
        inputs = [v, op32]
        op37 = forge.op.GreaterEqual(
            "op37",
            inputs[0],
            inputs[1],
        )

        # shapes: [(4, 34, 18, 1)] -> (4, 34, 18, 1)
        inputs = [self.get_constant("nconst9")]
        op38 = forge.op.Exp(
            "op38",
            inputs[0],
        )

        # shapes: [(4, 34, 18, 1)] -> (4, 34, 18, 1)
        inputs = [self.get_constant("nconst10")]
        op39 = forge.op.Exp(
            "op39",
            inputs[0],
        )

        # shapes: [(4, 17, 1, 63), (4, 17, 1, 63)] -> (4, 17, 1, 63)
        inputs = [self.get_constant("nconst11"), op35]
        op40 = forge.op.Divide(
            "op40",
            inputs[0],
            inputs[1],
        )

        # shapes: [(4, 34, 18, 12)] -> (4, 34, 18, 12)
        inputs = [self.get_constant("nconst12")]
        op41 = forge.op.Tanh(
            "op41",
            inputs[0],
        )

        # shapes: [(4, 34, 18, 1)] -> (4, 34, 18, 1)
        inputs = [op39]
        op42 = forge.op.Sigmoid(
            "op42",
            inputs[0],
        )

        # shapes: [(4, 1, 1, 63)] -> (4, 1, 1, 63)
        inputs = [self.get_constant("nconst13")]
        op43 = forge.op.Tanh(
            "op43",
            inputs[0],
        )

        # shapes: [(4, 34, 9, 63)] -> (4, 34, 9, 63)
        inputs = [self.get_constant("nconst14")]
        op44 = forge.op.Erf(
            "op44",
            inputs[0],
        )

        # shapes: [(4, 34, 18, 46), (4, 34, 18, 1), (4, 34, 18, 12), (4, 34, 18, 1), (4, 34, 18, 1), (4, 34, 18, 1), (4, 34, 18, 1)] -> (4, 34, 18, 63)
        inputs = [op30, op42, op41, op42, op38, self.get_constant("nconst15"), op34]
        op45 = forge.op.Concatenate(
            "op45", inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], axis=3
        )

        # shapes: [(4, 2, 1, 63), (4, 17, 1, 63), (4, 1, 1, 63), (4, 6, 1, 63), (4, 5, 1, 63), (4, 3, 1, 63)] -> (4, 34, 1, 63)
        inputs = [op37, op40, op43, op17, self.get_constant("nconst16"), op18]
        op46 = forge.op.Concatenate("op46", inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], axis=1)

        # shapes: [(4, 34, 9, 63)] -> (4, 34, 9, 63)
        inputs = [op44]
        op47 = forge.op.Tanh(
            "op47",
            inputs[0],
        )

        # shapes: [(4, 34, 18, 63)] -> (4, 34, 18, 63)
        inputs = [op45]
        op48 = forge.op.Abs(
            "op48",
            inputs[0],
        )

        # shapes: [(4, 34, 1, 63)] -> (4, 34, 1, 63)
        inputs = [op46]
        op49 = forge.op.LeakyRelu("op49", inputs[0], alpha=85.362414458034)

        # shapes: [(4, 34, 9, 63)] -> (4, 34, 9, 63)
        inputs = [op47]
        op50 = forge.op.Sigmoid(
            "op50",
            inputs[0],
        )

        # shapes: [(4, 34, 1, 63), (4, 34, 1, 63)] -> (4, 34, 1, 63)
        inputs = [op46, self.get_constant("nconst17")]
        op51 = forge.op.Divide(
            "op51",
            inputs[0],
            inputs[1],
        )

        # shapes: [(4, 34, 1, 63), (4, 34, 1, 63)] -> (4, 34, 1, 63)
        inputs = [op51, op46]
        v = forge.op.Max(
            "op52",
            inputs[0],
            inputs[1],
        )

        # shapes: [(4, 34, 1, 63)] -> (4, 34, 1, 63)
        inputs = [v]
        v = forge.op.Abs(
            "op53",
            inputs[0],
        )

        # shapes: [(4, 34, 1, 63), (4, 34, 18, 63), (4, 34, 9, 63), (4, 34, 1, 63), (4, 34, 1, 63), (4, 34, 1, 63), (4, 34, 1, 63)] -> (4, 34, 32, 63)
        inputs = [v, op48, op50, self.get_constant("nconst19"), self.get_constant("nconst18"), op51, op49]
        v = forge.op.Concatenate(
            "op54", inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], axis=2
        )

        return v


# @pytest.mark.xfail(reason="The model triggers a bug.")
def test_gen_model_1_403958(test_device):

    input_shapes = [
        (4, 17, 1, 63),
        (4, 17, 1, 63),
        (4, 34, 3, 1),
        (4, 34, 3, 1),
    ]
    model = GeneratedTestModel_1_403958("pytest_gen_model_healthy_forge_random_graph_algorithm_default_1_403958")

    verify_module(model, input_shapes, random_seed=403958)
