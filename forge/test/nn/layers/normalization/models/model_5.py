# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Test 5
# Test for Single Layer Layernorm
#

import torch

import forge
import forge.op
from forge.op import nn

from forge import ForgeModule, Tensor



class LayernormTest(ForgeModule):

    def __init__(
        self,
        input_shape,
        gamma_shape,
        beta_shape,
        dim,
        epsilon
    ):
        super().__init__("Test 5, Layernorm")

        self.testname = "Layernorm Test 5"
        self.input_shape = input_shape
        self.gamma_shape = gamma_shape
        self.beta_shape = beta_shape
        self.dim = dim
        self.epsilon = epsilon

        self.gamma = {
            f"gamma{i}": forge.Parameter(*self.gamma_shape, requires_grad=True)
            for i in range(1, 16)
        }
        self.beta = {
            f"beta{i}": forge.Parameter(*self.beta_shape, requires_grad=True)
            for i in range(1, 16)
        }
        self.train_param1 = forge.Parameter(*self.input_shape, requires_grad=True)
        self.train_param2 = forge.Parameter(*self.input_shape, requires_grad=True)
        self.train_param3 = forge.Parameter(*self.input_shape, requires_grad=True)

        self.inputs = []
        for i in range(1, 4):
            self.set_parameter(f"train_param{i}", torch.rand(*self.input_shape, requires_grad=True))
            self.inputs.append(Tensor.create_from_torch(torch.rand(*self.input_shape)))
        for i in range(1, 16):
            self.set_parameter(f"gamma{i}", torch.rand(*self.gamma_shape, requires_grad=True))
            self.set_parameter(f"beta{i}", torch.rand(*self.beta_shape, requires_grad=True))

    def forward(self, x1, x2, x3):

        # Layer 2
        mul1 = forge.op.Multiply("mul1", x1, self.train_param1)
        mul2 = forge.op.Multiply("mul2", x2, self.train_param2)
        mul3 = forge.op.Multiply("mul3", x3, self.train_param3)

        # Layer 3
        ln1 = nn.Layernorm(
            "ln1", 
            mul1, 
            self.gamma['gamma1'],
            self.beta['beta1'],
            self.dim,
            self.epsilon
        )
        ln2 = nn.Layernorm(
            "ln2", 
            mul2, 
            self.gamma['gamma2'], 
            self.beta['beta2'],
            self.dim, 
            self.epsilon
        )
        ln3 = nn.Layernorm(
            "ln3", 
            mul3, 
            self.gamma['gamma3'], 
            self.beta['beta3'], 
            self.dim, 
            self.epsilon
        )

        # Layer 4
        add1 = forge.op.Add("add1", ln1, x2)
        add2 = forge.op.Add("add2", ln2, x3)
        add3 = forge.op.Add("add3", ln3, self.train_param2)

        # Layer 5
        add4 = forge.op.Add("add4", add1, ln2)
        add5 = forge.op.Add("add5", add2, self.train_param1)
        ln4 = nn.Layernorm(
            "ln4", 
            add3, 
            self.gamma['gamma4'],
            self.beta['beta4'],
            self.dim, 
            self.epsilon
        )

        # Layer 6
        mul4 = forge.op.Multiply("mul4", ln1, add4)
        mul5 = forge.op.Multiply("mul5", x2, add5)
        add6 = forge.op.Add("add6", self.train_param2, ln4)

        # Layer 7
        ln5 = nn.Layernorm(
            "ln5", 
            mul4, 
            self.gamma['gamma5'],
            self.beta['beta5'],
            self.dim, 
            self.epsilon
        )
        ln6 = nn.Layernorm(
            "ln6", 
            mul5, 
            self.gamma['gamma6'],
            self.beta['beta6'],
            self.dim, 
            self.epsilon
        )
        ln7 = nn.Layernorm(
            "ln7", 
            add6, 
            self.gamma['gamma7'],
            self.beta['beta7'],
            self.dim, 
            self.epsilon
        )

        # Layer 8
        add7 = forge.op.Add("add7", add2, ln6)
        add8 = forge.op.Add("add8", add5, ln3)
        mul6 = forge.op.Multiply("mul6", ln5, mul5)
        mul7 = forge.op.Multiply("mul7", mul4, ln6)
        mul8 = forge.op.Multiply("mul8", mul5, ln7)

        # Layer 9
        ln8 = nn.Layernorm(
            "ln8", 
            mul6, 
            self.gamma['gamma8'],
            self.beta['beta8'],
            self.dim, 
            self.epsilon
        )
        ln9 = nn.Layernorm(
            "ln9", 
            add7, 
            self.gamma['gamma9'],
            self.beta['beta9'],
            self.dim, 
            self.epsilon
        )
        ln10 = nn.Layernorm(
            "ln10", 
            mul7, 
            self.gamma['gamma10'], 
            self.beta['beta10'], 
            self.dim, 
            self.epsilon
        )
        ln11 = nn.Layernorm(
            "ln11", 
            add8, 
            self.gamma['gamma11'],
            self.beta['beta11'],
            self.dim, 
            self.epsilon
        )
        ln12 = nn.Layernorm(
            "ln12",
            mul8,
            self.gamma['gamma12'], 
            self.beta['beta12'], 
            self.dim, 
            self.epsilon
        )

        # Layer 10
        mul9 = forge.op.Multiply("mul9", ln8, ln9)
        mul10 = forge.op.Multiply("mul10", ln10, ln11)

        # Layer 11
        ln13 = nn.Layernorm(
            "ln13", 
            mul9, 
            self.gamma['gamma13'], 
            self.beta['beta13'], 
            self.dim, 
            self.epsilon
        )
        ln14 = nn.Layernorm(
            "ln14", 
            mul10, 
            self.gamma['gamma14'], 
            self.beta['beta14'], 
            self.dim, 
            self.epsilon
        )

        # Layer 12
        add9 = forge.op.Add("add9", ln13, ln14)
        add10 = forge.op.Add("add10", ln14, ln12)

        # Layer 13
        mul11 = forge.op.Multiply("mul11", add9, add10)

        # Layer 14
        ln15 = nn.Layernorm(
            "ln15", 
            mul11, 
            self.gamma['gamma15'],
            self.beta['beta15'],
            self.dim, 
            self.epsilon
        )

        return ln15
