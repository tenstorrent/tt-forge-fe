# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Training optimizers
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from forge.compiled_graph_state import CompiledModel
from forge.tensor import Tensor
from forge.parameter import Parameter
import forge.torch_optimizers
from forge.torch_optimizers import AdamNoBiasCorrection


class Optimizer:
    """
    Optimizer base class
    """

    dynamic_params = False
    linked_modules: Optional[List[CompiledModel]] = None

    # For each parameter that we are optimizing, we store a dictionary of optimizer parameters for that particular parameter.
    # The derived classes will populate this dictionary with the necessary optimizer parameters.
    parameter_to_opt_inputs: Dict[str, Dict[str, Tensor]] = {}

    def __init__(self, parameters: Optional[List[Parameter]]):
        """
        Create baseline optimizer. If parameters are not passed at construction, they will be
        dynamically added during compilation.
        """
        self.dynamic_params = parameters is None

    def get_param_dict(self) -> Dict:
        """
        Return a dict of parameter node names and values to push to the device
        """
        raise RuntimeError("Subclasses should implement this.")

    # Get the list of optimizer parameters for each parameter that is being optimized.
    # E.g. for 'l1.weight' parameter, the optimizer parameters could be:
    # {
    #   ('l1.weight', 'lr'): 0.01,
    #   ('l1.weight', 'momentum'): 0.9,
    # }
    def get_optimizer_params(self) -> Dict[Tuple[str, str], Tensor]:
        opt_params = {}

        for parameter_name, params in self.parameter_to_opt_inputs.items():
            for opt_param_name, opt_param in params.items():
                opt_params[(parameter_name, opt_param_name)] = opt_param

        return opt_params

    def generate_op_trace(self, parameter, gradient):
        """
        Define the graph of ops involved in the optimizer eval.
        """
        raise RuntimeError("Subclasses should implement this.")

    def torch_parameter_update(
        self, *, parameter_name: str, parameter: torch.Tensor, gradient: torch.Tensor
    ) -> torch.Tensor:
        """
        Pytorch eval implementation for the optimizer

        Parameters
        ----------
        parameter : torch.Tensor
            parameter
        gradient : torch.Tensor
            gradient
        """
        raise RuntimeError("Subclasses should implement this.")

    def get_pytorch_optimizer(self, parameters: Dict[str, torch.Tensor]) -> torch.optim.Optimizer:
        raise RuntimeError("Subclasses should implement this.")

    def link_module(self, module: CompiledModel):
        if self.linked_modules is None:
            self.linked_modules = []
        self.linked_modules.append(module)

    def step(self):
        assert self.linked_modules is not None, "Optimizer must be linked to a module before calling step"
        for module in self.linked_modules:
            module.step()


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer

    Attributes
    ----------
    learning_rate : float
        learning_rate used by optimizer to adjust parameter
    parameter_to_opt_inputs : Dict[Parameter, Dict[str, Tensor]]
        Maps a Parameter with `requires_grad=True` to its associated
        optimizer_parameter_name -> Tensor dict.
    """

    def __init__(self, learning_rate: float, parameters: Optional[List[Parameter]] = None):
        super().__init__(parameters)
        self.learning_rate: float = learning_rate
        self.parameter_to_opt_inputs: Dict[str, Dict[str, Tensor]] = {}

        if parameters is not None:
            self.set_parameters_to_optimize(parameters)

    def set_parameters_to_optimize(self, parameters: List[Parameter]):
        # For each Parameter, we register its associated set of optimizer parameters
        for parameter in parameters:
            if parameter.requires_grad:
                self.parameter_to_opt_inputs[parameter.get_name()] = self.get_param_dict(parameter.pt_data_format)

    def get_param_dict(self, dtype: torch.dtype) -> Dict:
        """
        Return a dict of optimizer parameter names to tensor
        """
        # Forge needs a pytorch array for now
        # TODO(jchu): modify these two lines when we deprecate the old path
        learning_rate_torch = torch.full((1,), self.learning_rate, dtype=dtype)

        learning_rate = Tensor.create_from_torch(learning_rate_torch)
        return {"lr": learning_rate}

    def get_optimizer_state_keys(self) -> List:
        return []

    def get_type(self) -> str:
        return "sgd"

    def set_optimizer_parameters(self, learning_rate: Optional[float] = None) -> None:
        """
        Loop through every Parameter tensor with `requires_grad=True` and pushes
        `learning_rate` value to its associated optimizer parameter queue. By default,
        if no `learning_rate` is specified, the learning rate used to construct the
        sgd_optimizer is used.

        This method may be invoked multiple times as a way to
        adjust the updated optimizer parameter values across training loop iterations.

        Parameters
        ----------
        learning_rate : Optional[float]
            learning_rate

        Returns
        -------
        None

        """

        if learning_rate:
            self.learning_rate = learning_rate

        for parameter, opt_inputs in self.parameter_to_opt_inputs.items():
            learning_rate_tensor = torch.full((1,), self.learning_rate, dtype=opt_inputs["lr"].pt_data_format)
            opt_inputs["lr"] = Tensor.create_from_torch(learning_rate_tensor)

    def generate_op_trace(self, ac, parameter, gradient):
        lr = ac.input("lr", (1,))

        grad_times_lr = ac.op("multiply", (gradient, lr))
        param_minus_lr_times_grad = ac.op("subtract", (parameter, grad_times_lr))

        return param_minus_lr_times_grad

    def torch_parameter_update(
        self, *, parameter_name: str, parameter: torch.Tensor, gradient: torch.Tensor
    ) -> torch.Tensor:
        return parameter - self.learning_rate * gradient

    def get_pytorch_optimizer(self, parameters: Dict[str, torch.Tensor], lr=None) -> torch.optim.Optimizer:
        """
        Return an equivalent pytorch optimizer, used for verification.
        """
        if lr is not None:
            self.set_optimizer_parameters(learning_rate=lr)
        return torch.optim.SGD([p for p in parameters.values()], self.learning_rate)


class Adam(Optimizer):
    """
    Adam Optimizer

    Attributes
    ----------
    learning_rate : float
        learning_rate used by optimizer to adjust parameter
    parameter_to_opt_inputs : Dict[Parameter, Dict[str, Tensor]]
        Maps a Parameter with `requires_grad=True` to its associated
        optimizer_parameter_name -> Tensor dict.
    betas : (Tuple[float, float], optional)
        coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
    eps : (float, optional)
        term added to the denominator to improve numerical stability (default: 1e-8)
    weight_decay : (float, optional)
        weight decay (L2 penalty) (default: 0)
    bias_correction: (bool, optional)
        use bias correction
    """

    def __init__(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        bias_correction: bool = True,
        parameters: Optional[List[Parameter]] = None,
        enable_adam_w: bool = False,
    ):
        super().__init__(parameters)
        # optimizer constants
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.bias_correction = bias_correction
        self.torch_optimizer = None

        self.learning_rate = learning_rate
        self.parameter_to_opt_inputs: Dict[str, Dict[str, Tensor]] = {}
        self.parameter_to_opt_torch_inputs: Dict[str, Dict[str, Tensor]] = {}
        self.enable_adam_w = enable_adam_w

        if parameters:
            self.set_parameters_to_optimize(parameters)

    def get_cpu_param_dict(self, dtype: torch.dtype, shape: Tuple[int]) -> Dict:
        # TODO: shapes are set to the shape of the parameter because of the following issue
        # https://github.com/tenstorrent/tt-metal/issues/16352
        if self.bias_correction:
            return {
                "torch_mean": torch.full(shape, 0.0, dtype=dtype),
                "torch_variance": torch.full(shape, 0.0, dtype=dtype),
                "torch_beta1_pow": torch.full(shape, 1.0, dtype=dtype),
                "torch_beta2_pow": torch.full(shape, 1.0, dtype=dtype),
            }
        else:
            return {
                "torch_mean": torch.full(shape, 0.0, dtype=dtype),
                "torch_variance": torch.full(shape, 0.0, dtype=dtype),
            }

    def get_param_dict(self, dtype: torch.dtype, shape: Tuple[int]) -> Dict:
        """
        Return a dict of optimizer parameter names to tensor
        """
        torch_lr = torch.full((1,), self.learning_rate, dtype=dtype)
        if self.bias_correction:
            return {
                "lr": Tensor.create_from_torch(torch.full(shape, self.learning_rate, dtype=dtype)),
                "mean": Tensor.create_from_torch(torch.full(shape, 0.0, dtype=dtype)),
                "variance": Tensor.create_from_torch(torch.full(shape, 0.0, dtype=dtype)),
                "beta1_pow": Tensor.create_from_torch(torch.full(shape, 1.0, dtype=dtype)),
                "beta2_pow": Tensor.create_from_torch(torch.full(shape, 1.0, dtype=dtype)),
            }
        else:
            return {
                "lr": Tensor.create_from_torch(torch.full(shape, self.learning_rate, dtype=dtype)),
                "mean": Tensor.create_from_torch(torch.full(shape, 0.0, dtype=dtype)),
                "variance": Tensor.create_from_torch(torch.full(shape, 0.0, dtype=dtype)),
            }

    def get_optimizer_state_keys(self) -> List:
        if self.bias_correction:
            return ["mean", "variance", "beta1_pow", "beta2_pow"]
        else:
            return ["mean", "variance"]

    def get_type(self) -> str:
        return "adam"

    def set_parameters_to_optimize(self, parameters: List[Parameter]):
        # For each Parameter, we register its associated set of optimizer parameters
        for parameter in parameters:
            if parameter.requires_grad:
                self.parameter_to_opt_inputs[parameter.get_name()] = self.get_param_dict(
                    parameter.pt_data_format, parameter.shape.get_pytorch_shape()
                )
                self.parameter_to_opt_torch_inputs[parameter.get_name()] = self.get_cpu_param_dict(
                    parameter.pt_data_format, parameter.shape.get_pytorch_shape()
                )

    def set_optimizer_parameters(self, learning_rate: Optional[float] = None) -> None:
        """
        Loop through every Parameter tensor with `requires_grad=True` and pushes
        `learning_rate` value to its associated optimizer parameter queue. By default,
        if no `learning_rate` is specified, the learning rate used to construct the
        sgd_optimizer is used.

        This method may be invoked multiple times as a way to
        adjust the updated optimizer parameter values across training loop iterations.

        Parameters
        ----------
        learning_rate : Optional[float]
            learning_rate

        Returns
        -------
        None

        """

        if learning_rate:
            self.learning_rate = learning_rate

        # {mean, variance} get updated in the loopback
        for parameter, opt_inputs in self.parameter_to_opt_inputs.items():
            torch_lr = torch.full(parameter.shape.as_list(), self.learning_rate, dtype=opt_inputs["lr"].pt_data_format)
            opt_inputs["lr"] = Tensor.create_from_torch(torch_lr)

    def generate_op_trace(self, ac, parameter, gradient):

        parameter_shape = parameter.shape.as_list()
        if self.weight_decay > 0.0:
            weight_decay = ac.tensor(torch.full(parameter_shape, self.weight_decay))
        else:
            weight_decay = None

        if weight_decay and not self.enable_adam_w:
            weight_decay_times_param = ac.op("multiply", (weight_decay, parameter))
            gradient = ac.op("add", (gradient, weight_decay_times_param))

        # self.mean = self.beta1 * self.mean + one_minus_beta1 * gradient
        mean = ac.input("mean", parameter.shape, copy_consteval_operations=True)
        beta1 = ac.tensor(torch.full(parameter_shape, self.beta1))
        one_minus_beta1 = ac.tensor(torch.full(parameter_shape, 1 - self.beta1))
        mean_times_beta1 = ac.op("multiply", (mean, beta1))
        gradient_times_one_minus_beta1 = ac.op("multiply", (gradient, one_minus_beta1))
        updated_mean = ac.op("add", (mean_times_beta1, gradient_times_one_minus_beta1))

        # self.variance = self.beta2 * self.variance + one_minus_beta2 * gradient**2
        variance = ac.input("variance", parameter.shape, copy_consteval_operations=True)
        beta2 = ac.tensor(torch.full(parameter_shape, self.beta2))
        one_minus_beta2 = ac.tensor(torch.full(parameter_shape, 1 - self.beta2))
        variance_times_beta2 = ac.op("multiply", (variance, beta2))
        gradient_squared = ac.op("multiply", (gradient, gradient))
        gradient_squared_times_one_minus_beta2 = ac.op("multiply", (gradient_squared, one_minus_beta2))
        updated_variance = ac.op("add", (variance_times_beta2, gradient_squared_times_one_minus_beta2))
        if self.bias_correction:
            # bias_correction1 = 1 - beta1 ** step
            beta1_one = ac.tensor(torch.full(parameter_shape, 1.0))
            beta1_pow = ac.input("beta1_pow", parameter.shape, disable_consteval=True)  # stores beta1 ** step
            updated_beta1_pow = ac.op("multiply", (beta1_pow, beta1))
            bias_correction1 = ac.op("subtract", (beta1_one, updated_beta1_pow))
            reciprocal_bias_correction1 = ac.op("reciprocal", (bias_correction1,))

            # bias_correction2 = 1 - beta2 ** step
            beta2_one = ac.tensor(torch.full(parameter_shape, 1.0))
            beta2_pow = ac.input("beta2_pow", parameter.shape, disable_consteval=True)  # stores beta2 ** step
            updated_beta2_pow = ac.op("multiply", (beta2_pow, beta2))
            bias_correction2 = ac.op("subtract", (beta2_one, updated_beta2_pow))
            sqrt_bias_correction2 = ac.op("sqrt", (bias_correction2,))
            reciprocal_sqrt_bias_correction2 = ac.op("reciprocal", (sqrt_bias_correction2,))

            # sqrt_of_variance / sqrt_bias_correction2
            sqrt_of_variance_biased = ac.op("sqrt", (updated_variance,))
            sqrt_of_variance = ac.op("multiply", (sqrt_of_variance_biased, reciprocal_sqrt_bias_correction2))
        else:
            sqrt_of_variance = ac.op("sqrt", (updated_variance,))

        epsilon = ac.tensor(torch.full(parameter_shape, self.epsilon))
        sqrt_of_variance_plus_epsilon = ac.op("add", (sqrt_of_variance, epsilon))
        reciprocal_of_sqrt_of_variance_plus_epsilon = ac.op("reciprocal", (sqrt_of_variance_plus_epsilon,))

        if self.bias_correction:
            # mean / bias_correction1
            updated_mean_unbiased = ac.op("multiply", (updated_mean, reciprocal_bias_correction1))
            mean_times_reciprocal_of_sqrt_of_variance_plus_epsilon = ac.op(
                "multiply", (updated_mean_unbiased, reciprocal_of_sqrt_of_variance_plus_epsilon)
            )
        else:
            mean_times_reciprocal_of_sqrt_of_variance_plus_epsilon = ac.op(
                "multiply", (updated_mean, reciprocal_of_sqrt_of_variance_plus_epsilon)
            )

        if weight_decay and self.enable_adam_w:
            # weight_decay * param + mean/sqrt(var)
            weight_decay_times_param = ac.op("multiply", (weight_decay, parameter))
            mean_times_reciprocal_of_sqrt_of_variance_plus_epsilon_plus_weight_decay_times_param = ac.op(
                "add", (mean_times_reciprocal_of_sqrt_of_variance_plus_epsilon, weight_decay_times_param)
            )
        else:
            mean_times_reciprocal_of_sqrt_of_variance_plus_epsilon_plus_weight_decay_times_param = (
                mean_times_reciprocal_of_sqrt_of_variance_plus_epsilon
            )

        lr = ac.input("lr", parameter.shape)
        parameter_delta = ac.op(
            "multiply", (mean_times_reciprocal_of_sqrt_of_variance_plus_epsilon_plus_weight_decay_times_param, lr)
        )
        updated_parameter = ac.op("subtract", (parameter, parameter_delta))

        # in the old spatial1, there was a loopback for each of {updated_mean, updated_variance}
        ac.loopback(updated_mean, mean)
        ac.loopback(updated_variance, variance)
        if self.bias_correction:
            ac.loopback(updated_beta1_pow, beta1_pow)
            ac.loopback(updated_beta2_pow, beta2_pow)
        return updated_parameter

    def torch_parameter_update(
        self, *, parameter_name: str, parameter: torch.Tensor, gradient: torch.Tensor
    ) -> torch.Tensor:
        if not self.enable_adam_w and self.weight_decay > 0.0:
            gradient = gradient + self.weight_decay * parameter

        if self.bias_correction:
            updated_torch_mean = (
                self.beta1 * self.parameter_to_opt_torch_inputs[parameter_name]["torch_mean"]
                + (1 - self.beta1) * gradient
            ) / (1 - self.beta1)
            updated_torch_variance = (
                self.beta2 * self.parameter_to_opt_torch_inputs[parameter_name]["torch_variance"]
                + (1 - self.beta2) * gradient**2
            ) / (1 - self.beta2)
        else:
            updated_torch_mean = (
                self.beta1 * self.parameter_to_opt_torch_inputs[parameter_name]["torch_mean"]
                + (1 - self.beta1) * gradient
            )
            updated_torch_variance = (
                self.beta2 * self.parameter_to_opt_torch_inputs[parameter_name]["torch_variance"]
                + (1 - self.beta2) * gradient**2
            )

        updated_parameter = parameter - self.learning_rate * (
            updated_torch_mean / (torch.sqrt(updated_torch_variance) + self.epsilon)
            + (self.weight_decay * parameter if self.enable_adam_w and self.weight_decay > 0.0 else 0)
        )

        return updated_parameter

    def get_pytorch_optimizer(self, parameters: Dict[str, torch.Tensor], lr=None) -> torch.optim.Optimizer:
        """
        Return an equivalent pytorch optimizer, used for verification.
        """
        if lr is not None:
            self.set_optimizer_parameters(learning_rate=lr)
        # May want to initialize with initial learning_rate
        if not self.torch_optimizer:
            if self.bias_correction:
                optim = torch.optim.AdamW if self.enable_adam_w else torch.optim.Adam
                self.torch_optimizer = optim(
                    [p for p in parameters.values()],
                    lr=self.learning_rate,
                    betas=(self.beta1, self.beta2),
                    eps=self.epsilon,
                    weight_decay=self.weight_decay,
                )
            else:
                self.torch_optimizer = AdamNoBiasCorrection(
                    [p for p in parameters.values()],
                    lr=self.learning_rate,
                    betas=(self.beta1, self.beta2),
                    eps=self.epsilon,
                    weight_decay=self.weight_decay,
                    enable_adam_w=self.enable_adam_w,
                )
        return self.torch_optimizer


class AdamW(Adam):
    """
    Implements weighted Adam optimizer.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        bias_correction: bool = True,
        parameters: Optional[List[Parameter]] = None,
    ):
        super().__init__(
            learning_rate,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction,
            parameters,
            enable_adam_w=True,
        )
