# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod

import torch

from forge.verify.compare import compare_with_golden, compute_required_tolerances


class ValueChecker(ABC):
    """Abstract base class for value checking strategies."""

    def __init__(self, rtol: float = 1e-05, atol: float = 1e-08):
        self.rtol = rtol
        self.atol = atol

    @abstractmethod
    def check(self, fw_out, co_out):
        """Abstract method for performing the value check between two tensors."""
        pass


class AutomaticValueChecker(ValueChecker):
    """Checks values automatically using PCC, all_close or rogerstanimoto based on type of the tensor.
    if tensor is scalar all_close is used, if tensor is boolean based rogerstanimoto is used, else PCC is used."""

    def __init__(
        self, pcc: float = 0.99, rtol: float = 1e-05, atol: float = 1e-08, dissimilarity_threshold: float = 1e-03
    ):
        super().__init__(rtol, atol)
        self.pcc = pcc
        self.dissimilarity_threshold = (
            dissimilarity_threshold  # threshold picked empirically. We will update it as TTNN evolves
        )

    def check(self, fw_out, co_out):
        if not compare_with_golden(fw_out, co_out, self.pcc, self.rtol, self.atol, self.dissimilarity_threshold):
            raise ValueError(
                f"Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model={fw_out}, compiled_model={co_out}"
            )


class AllCloseValueChecker(ValueChecker):
    """Checks values using torch.all_close."""

    def check(self, fw_out, co_out):
        assert fw_out.dtype not in [
            torch.int32,
            torch.int64,
            torch.bool,
        ], f"AllCloseValueChecker (all_close): all_close doesn't make sense for integer/bool types"

        if not torch.allclose(fw_out, co_out, rtol=self.rtol, atol=self.atol):
            atol, rtol = compute_required_tolerances(fw_out, co_out)
            raise ValueError(
                f"Data mismatch -> AllCloseValueChecker (all_close):\n"
                f"- Tensor mismatch. Required rtol={self.rtol}, atol={self.atol}\n"
                f"- Observed maximum relative diff: {rtol}, maximum absolute diff: {atol}\n"
                f"- Framework output: ({fw_out.shape})\n"
                f"{fw_out}\n"
                f"- Compiled model output: ({co_out.shape})\n"
                f"{co_out}"
            )


class FullValueChecker(ValueChecker):
    """Performs PCC and all_close for float tensors."""

    def __init__(
        self, pcc: float = 0.99, rtol: float = 1e-05, atol: float = 1e-08, dissimilarity_threshold: float = 1e-03
    ):
        super().__init__(rtol, atol)
        self.pcc = pcc
        self.dissimilarity_threshold = (
            dissimilarity_threshold  # threshold picked empirically. We will update it as TTNN evolves
        )

    def check(self, fw_out, co_out):
        if not compare_with_golden(fw_out, co_out, self.pcc, self.rtol, self.atol, self.dissimilarity_threshold):
            raise ValueError(
                f"Data mismatch -> FullChecker (compare_with_golden): framework_model={fw_out}, compiled_model={co_out}"
            )

        all_close_check = True
        if fw_out.dtype not in [
            torch.int32,
            torch.int64,
            torch.bool,
        ]:  # allclose doesn't make sense for integer/bool types
            all_close_check = torch.allclose(fw_out, co_out, rtol=self.rtol, atol=self.atol)

        if not all_close_check:
            atol, rtol = compute_required_tolerances(fw_out, co_out)
            raise ValueError(
                f"Data mismatch -> FullChecker (all_close):\n"
                f"- Tensor mismatch. Required rtol={self.rtol}, atol={self.atol}\n"
                f"- Observed maximum relative diff: {rtol}, maximum absolute diff: {atol}\n"
                f"- Framework output: ({fw_out.shape})\n"
                f"{fw_out}\n"
                f"- Compiled model output: ({co_out.shape})\n"
                f"{co_out}"
            )
