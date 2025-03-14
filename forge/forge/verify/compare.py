# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


from typing import Union
import os

import torch
import tensorflow as tf
import numpy as np
from loguru import logger
from scipy.spatial import distance
from typing import Union, Tuple, List, Optional

from forge.tensor import narrow_forge_tensor_to_pytorch

# Compares golden and calculated tensors. Using allclose for scalar values, rogerstanimoto for bool tensors, pcc otherwise
def compare_with_golden(
    golden: Union[torch.Tensor, tf.Tensor, tf.Variable],
    calculated: torch.Tensor,
    pcc: float = 0.99,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    dissimilarity_threshold: float = 1e-03,
):
    if golden.dtype == torch.bool:
        result = compare_with_golden_bool(golden, calculated, dissimilarity_threshold)
    if golden.flatten().size() != (1,):  # PCC for single values doesn't work
        result = compare_with_golden_pcc(golden, calculated, pcc)
    else:
        # For scalar values, we can't calculate PCC, but we can compare golden and calculated values using relative and absolute tolerances
        golden = golden.flatten()[0]
        calculated = calculated.flatten()[0]

        all_close = torch.allclose(golden, calculated, rtol=rtol, atol=atol)
        if not all_close:
            req_atol, req_rtol = compute_required_tolerances(golden, calculated)
            logger.error("Tensor mismatch. Required rtol={}, atol={}", rtol, atol)
            logger.error("Observed maximum relative diff: {}, maximum absolute diff: {}", req_rtol, req_atol)
            logger.error("Golden: (shape = {}", golden.shape)
            logger.error(golden)
            logger.error("Calculated: (shape = {}", calculated.shape)
            logger.error(calculated)

        result = all_close

    return result


# Calculates pcc between golden and calculated tensors. If calculated pcc is >= than pcc threshold, returns True
def compare_with_golden_pcc(
    golden: Union[torch.Tensor, tf.Tensor, tf.Variable],
    calculated: torch.Tensor,
    pcc: float = 0.99,
):
    assert pcc >= 0, "PCC threshold must be >= 0"
    assert golden.flatten().size() != (1,), "PCC for single values doesn't work"

    pcc_value = calculate_pcc(golden, calculated)
    if pcc_value >= pcc:
        logger.trace("PCC is correct")
        logger.trace("Golden: (shape = {}", golden.shape)
        logger.trace(golden)
        logger.trace("Calculated: (shape = {}", calculated.shape)
        logger.trace(calculated)
        return True
    else:
        logger.error("Tensor mismatch. PCC = {}, but required = {}", pcc_value, pcc)
        logger.trace("Golden: (shape = {}", golden.shape)
        logger.trace(golden)
        logger.trace("Calculated: (shape = {}", calculated.shape)
        logger.trace(calculated)
        return False


def compare_with_golden_bool(
    golden: Union[torch.Tensor, tf.Tensor, tf.Variable],
    calculated: torch.Tensor,
    dissimilarity_threshold: float = 1e-03,  # threshold picked empirically. We will update it as TTNN evolves
):
    if calculated.dtype != torch.bool:
        calculated = calculated.to(torch.bool)

    golden_squeezed = golden.view(-1).detach().numpy()
    calculated_squeezed = calculated.view(-1).detach().numpy()
    dissimilarity = distance.rogerstanimoto(golden_squeezed, calculated_squeezed)

    if dissimilarity <= dissimilarity_threshold:
        logger.trace("Bool vectors are not dissimilar. Dissimiliarity = {}", dissimilarity)
        logger.trace("Golden: (shape = {}", golden.shape)
        logger.trace(golden)
        logger.trace("Calculated: (shape = {}", calculated.shape)
        logger.trace(calculated)
        return True
    else:
        logger.error("Tensor mismatch. Dissimiliarity = {}", dissimilarity)
        return False


def calculate_pcc(a, b):
    if torch.all(torch.isnan(a)) and torch.all(torch.isnan(b)):
        logger.warning("Both tensors are 'nan'")
        return 1.0

    if torch.all(torch.isnan(a)) or torch.all(torch.isnan(b)):
        logger.error("One tensor is all nan, the other is not.")
        return 0.0

    # Test if either is completely zero
    if torch.any(a.bool()) != torch.any(b.bool()):
        return 0.0

    # if torch.any(torch.isinf(a)) or torch.any(torch.isinf(b)):
    #    raise RuntimeError(f"Tensor overflow to infinity: \n{a}\n{b}")

    # if torch.any(torch.isneginf(a)) or torch.any(torch.isneginf(b)):
    #    raise RuntimeError(f"Tensor overflow to negative infinity: \n{a}\n{b}")

    # For now, mask all infs and nans so that we check the rest... TODO
    a = a.clone()
    a[torch.logical_or(torch.isnan(a), torch.logical_or(torch.isinf(a), torch.isneginf(a)))] = 0
    b = b.clone()
    b[torch.logical_or(torch.isnan(b), torch.logical_or(torch.isinf(b), torch.isneginf(b)))] = 0

    if torch.equal(a, b):
        return 1.0

    if a.dtype == torch.bfloat16:
        a = a.type(torch.float32)
        b = b.type(torch.float32)
    pcc = np.min(
        np.ma.corrcoef(
            np.ma.masked_invalid(torch.squeeze(a).detach().numpy()).flatten(),
            np.ma.masked_invalid(torch.squeeze(b).detach().numpy()).flatten(),
        )
    )

    if isinstance(pcc, np.ma.core.MaskedConstant):
        return 1.0

    return pcc


# Deprecated: avoid using it, instead use compare_with_golden
def compare_tensor_to_golden(
    name: str,
    golden: Union[torch.Tensor, tf.Tensor, tf.Variable],
    calculated: torch.Tensor,
    is_forge=False,
    rtol=None,
    atol=None,
    pcc=None,
    warning_only=False,
    relative_atol=None,
    verify_cfg=None,
):
    # Convert golden to pytorch tensor for comparisons
    if isinstance(golden, (tf.Tensor, tf.Variable)):
        golden = torch.from_numpy(golden.numpy())

    if golden.dtype == torch.bool and calculated.dtype != torch.bool:
        calculated = calculated.to(torch.bool)

    if golden.dtype == torch.bool and calculated.dtype == torch.bool:
        return bool(torch.all(golden == calculated))

    if is_forge:
        calculated = narrow_forge_tensor_to_pytorch(calculated, golden.shape)

    if rtol is None or (isinstance(rtol, dict) and (golden.dtype not in rtol or rtol[golden.dtype] is None)):
        if verify_cfg is not None and golden.dtype in verify_cfg.rtol and verify_cfg.rtol[golden.dtype] is not None:
            rtol = verify_cfg.rtol[golden.dtype]
        else:
            rtol = 0  # use atol only, rtol is unreliable for very small numbers
    elif isinstance(rtol, dict):
        rtol = rtol[golden.dtype]

    if atol is None or (isinstance(atol, dict) and (golden.dtype not in atol or atol[golden.dtype] is None)):
        if verify_cfg is not None and golden.dtype in verify_cfg.atol and verify_cfg.atol[golden.dtype] is not None:
            atol = verify_cfg.atol[golden.dtype]
        else:
            if relative_atol is None and verify_cfg is not None:
                relative_atol = verify_cfg.relative_atol
            if relative_atol is None:
                relative_atol = 0.1

            if torch.all(torch.isnan(golden)):
                atol = 0
            else:
                max_value = (torch.max(torch.abs(golden[~torch.isnan(golden)]))).item()
                atol = max_value * relative_atol  # allow up to 'relative_atol' error
    elif isinstance(atol, dict):
        atol = atol[golden.dtype]

    if pcc is None and verify_cfg is not None:
        pcc = verify_cfg.pcc

    while len(calculated.shape) > len(golden.shape) and calculated.shape[0] == 1:
        calculated = calculated.squeeze(0)

    while len(golden.shape) > len(calculated.shape) and golden.shape[0] == 1:
        golden = golden.squeeze(0)

    if not golden.shape == calculated.shape:
        logger.error("Tensor shape mismatch on {}", name)
        logger.debug("Golden: (shape = {}", golden.shape)
        logger.debug("Calculated: (shape = {}", calculated.shape)
        return False

    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    ok = torch.allclose(golden, calculated, rtol=rtol, atol=atol, equal_nan=True)
    callback_ok = (
        True
        if verify_cfg is None or verify_cfg.golden_compare_callback is None
        else verify_cfg.golden_compare_callback(golden, calculated)
    )
    ok &= callback_ok
    pcc_value = 0
    if not (pcc is None or golden.flatten().size() == (1,)):  # PCC for single values doesn't work
        pcc_value = calculate_pcc(golden, calculated)
        if pcc_value >= pcc and not ok:
            logger.warning("PCC is correct but allclose failed on {}", name)
            logger.trace("Golden: (shape = {}", golden.shape)
            logger.trace(golden)
            logger.trace("Calculated: (shape = {}", calculated.shape)
            logger.trace(calculated)
            logger.warning(
                "Max ATOL Delta: "
                + "{:.3e}".format(torch.max(torch.abs(golden - calculated)).item())
                + ", atol="
                + "{}".format(atol)
            )
            logger.warning(
                "Max RTOL Delta: "
                + "{:.3e}".format(torch.max(torch.abs(golden - calculated) / calculated).item())
                + ", rtol="
                + "{}".format(rtol)
            )
        ok |= pcc_value >= pcc

    if not ok:
        if warning_only:
            logger.warning("Tensor mismatch on {}", name)
        else:
            logger.error("Tensor mismatch on {}", name)
        logger.trace("Golden: (shape = {}", golden.shape)
        logger.trace(golden)
        logger.trace("Calculated: (shape = {}", calculated.shape)
        logger.trace(calculated)
        logger.info(
            "Max ATOL Delta: "
            + "{:.3e}".format(torch.max(torch.abs(golden - calculated)).item())
            + ", atol="
            + "{}".format(atol)
        )
        logger.info(
            "Max RTOL Delta: "
            + "{:.3e}".format(torch.max(torch.abs(golden - calculated) / calculated).item())
            + ", rtol="
            + "{}".format(rtol)
        )
        if pcc is not None:
            logger.info("PCC got={}, required={}", pcc_value, pcc)
        if not callback_ok:
            logger.info("User golden_compare_callback returned False")
        # torch.set_printoptions(profile="full")
        # print(golden-calculated)
        # torch.set_printoptions(profile="default")
        if not warning_only:
            return False
    else:
        if os.environ.get("SHOW_MATCHING", "0") != "0":
            logger.trace("Golden: (shape = {}", golden.shape)
            logger.trace(golden)
            logger.trace("Calculated (correct): (shape = {}", calculated.shape)
            logger.trace(calculated)
            if pcc is not None:
                logger.debug("PCC (correct) got={}, required={}", pcc_value, pcc)
        logger.debug("Tensors match on {}", name)

    return True


def compute_required_tolerances(a, b):
    if a.shape != b.shape:
        raise ValueError("Tensors must have the same shape")

    diff = torch.abs(a - b)
    abs_b = torch.abs(b)

    # Compute the required atol (assuming rtol=0)
    required_atol = torch.max(diff)

    # Compute the required rtol (assuming atol=0)
    # Avoid division by zero by setting a high rtol for zero values in b
    safe_abs_b = torch.where(abs_b == 0, torch.tensor(float("inf")), abs_b)
    required_rtol = torch.max(diff / safe_abs_b)

    return required_atol.item(), required_rtol.item()


def prepare_tensors(golden, calculated):
    # Convert boolean tensors to float; so ATOL can be calculated.
    if golden.dtype == torch.bool:
        golden = golden.to(torch.float)

    # TTNN does not support all the data types. So convert 'ret' tensor type to
    # match 'golden' tensor type.
    if golden.dtype != calculated.dtype:
        calculated = calculated.to(golden.dtype)

    return golden, calculated


def calculate_atol(golden, calculated):
    if torch.equal(golden, calculated):
        return 0.0

    golden, calculated = prepare_tensors(golden, calculated)

    # Handle NaN and Inf by verifying if NaN and Inf exists at same location in
    # both tensors.
    golden_nan_mask = torch.isnan(golden)
    calculated_nan_mask = torch.isnan(calculated)
    golden_inf_mask = torch.isinf(golden)
    calculated_inf_mask = torch.isinf(calculated)

    # Compare NaN values (NaN == NaN is considered True).
    if not torch.all(golden_nan_mask == calculated_nan_mask):
        return torch.nan

    # Compare Inf values (Inf == Inf is considered True).
    if not torch.all(golden_inf_mask == calculated_inf_mask):
        return torch.inf

    # Verify if respective Inf values in both tensors have same sign.
    golden_sign = torch.sign(golden)
    calculated_sign = torch.sign(calculated)
    sign_comparison = golden_sign == calculated_sign
    masked_sign_comparison = torch.where(calculated_inf_mask, sign_comparison, torch.tensor(True))
    if not torch.all(masked_sign_comparison):
        return torch.inf

    # Replace NaN values with 0 to avoid having NaN as ATOL
    golden[golden_nan_mask] = 0
    calculated[calculated_nan_mask] = 0

    # Replace Inf values with 0 to avoid having NaN as ATOL
    golden[golden_inf_mask] = 0
    calculated[calculated_inf_mask] = 0

    return torch.max(torch.abs(golden - calculated)).item()


def compute_pcc_and_atol(
    framework_outputs: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], compiled_outputs: List[torch.Tensor]
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute the Pearson correlation coefficient (PCC) and absolute tolerance (ATOL)
    between corresponding framework (golden) and compiled outputs.

    Parameters:
        framework_outputs: Tuple or list of torch.Tensor representing the expected outputs.
        compiled_outputs: List of torch.Tensor representing the computed outputs.

    Returns:
        A tuple (calculated_pcc, calculated_atol) where:
            - calculated_pcc is the average PCC computed over non-scalar tensors.
              If only scalar outputs are present, this is set to None.
            - calculated_atol is the average absolute tolerance computed over all outputs.
        If the number of framework and compiled outputs do not match, returns (None, None).
    """
    if len(framework_outputs) != len(compiled_outputs):
        logger.info(
            f"Input count mismatch: framework_outputs={len(framework_outputs)}, compiled_outputs={len(compiled_outputs)}"
        )
        return None, None

    pcc_values = []
    atol_values = []

    for idx, (fw_out, co_out) in enumerate(zip(framework_outputs, compiled_outputs), start=1):
        # For non-scalar tensors, ensure the shapes match
        if fw_out.numel() != 1 and fw_out.shape != co_out.shape:
            logger.info(f"Tensor {idx} - Shape mismatch: fw_out shape={fw_out.shape}, co_out shape={co_out.shape}")
            continue

        # If the output is a scalar, compute only ATOL
        if fw_out.numel() == 1:
            logger.info(f"Tensor {idx} - fw_out is scalar")
            atol = calculate_atol(fw_out, co_out)
            atol_values.append(atol)
        else:
            golden, calculated = prepare_tensors(fw_out, co_out)
            pcc = calculate_pcc(golden, calculated)
            pcc_values.append(pcc)
            atol = calculate_atol(fw_out, co_out)
            atol_values.append(atol)

    calculated_pcc = None if len(pcc_values) == 0 else sum(pcc_values) / len(pcc_values)
    calculated_atol = sum(atol_values) / len(atol_values) if atol_values else None

    return calculated_pcc, calculated_atol
