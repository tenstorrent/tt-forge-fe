# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


from multiprocessing.pool import ThreadPool
import os
from typing import Union

import torch
import tensorflow as tf
import numpy as np
from loguru import logger
from scipy.spatial import distance
from typing import Union, Tuple, List, Optional
from forge.verify.utils import convert_to_supported_pytorch_dtype

# Compares golden and calculated tensors. Using allclose for scalar values, rogerstanimoto for bool tensors, pcc otherwise
def compare_with_golden(
    golden: Union[torch.Tensor, tf.Tensor, tf.Variable],
    calculated: torch.Tensor,
    pcc: float = 0.99,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    dissimilarity_threshold: float = 1e-03,  # threshold picked empirically. We will update it as TTNN evolves
):

    if golden.dtype == torch.bool:
        calculated_dissimilarity = calculate_dissimilarity(golden, calculated)
        result = compare_dissimilarity(calculated_dissimilarity, dissimilarity_threshold)
    elif golden.flatten().size() != (1,):  # PCC for single values doesn't work
        calculated_pcc = calculate_pcc(golden, calculated)
        result = compare_pcc(calculated_pcc, pcc)
    else:
        # For scalar values, we can't calculate PCC, but we can compare golden and calculated values using relative and absolute tolerances
        golden = golden.flatten()[0]
        calculated = calculated.flatten()[0]

        all_close = torch.allclose(golden, calculated, rtol=rtol, atol=atol)
        if not all_close:
            req_atol, req_rtol = compute_required_tolerances(golden, calculated)
            logger.error("Tensor mismatch. Required rtol={}, atol={}", rtol, atol)
            logger.error("Observed maximum relative diff: {}, maximum absolute diff: {}", req_rtol, req_atol)

        result = all_close

    if not result:
        logger.error("Golden: (shape = {}", golden.shape)
        logger.error(golden)
        logger.error("Calculated: (shape = {}", calculated.shape)
        logger.error(calculated)

    return result


def calculate_dissimilarity(golden: torch.Tensor, calculated: torch.Tensor):
    if calculated.dtype != torch.bool:
        calculated = calculated.to(torch.bool)

    golden_squeezed = golden.reshape(-1).detach().numpy()
    calculated_squeezed = calculated.reshape(-1).detach().numpy()
    dissimilarity = distance.rogerstanimoto(golden_squeezed, calculated_squeezed)
    return dissimilarity


def compare_dissimilarity(calculated_dissimilarity: float, dissimilarity_threshold: float = 1e-03):
    if calculated_dissimilarity <= dissimilarity_threshold:
        logger.trace("Bool vectors are not dissimilar. Dissimiliarity = {}", calculated_dissimilarity)
        return True
    else:
        logger.error(
            "Tensor mismatch calculated dissimiliarity = {} dissimilarity_threshold = {}",
            calculated_dissimilarity,
            dissimilarity_threshold,
        )
        return False


# Helper function to calculate PCC over a chunk of rows
def calc_chunk_pcc(a, b, chunk_size, chunk_start):
    a_chunk = a[chunk_start : chunk_start + chunk_size]
    b_chunk = b[chunk_start : chunk_start + chunk_size]

    chunk_pcc = np.min(np.ma.corrcoef(a_chunk, b_chunk))

    return chunk_pcc


def calculate_or_estimate_pcc(
    a: torch.Tensor, b: torch.Tensor, tensor_size_threshold: int, chunk_size: int
) -> np.float64:

    """
    Calculate (estimate) Pearson Correlation Coefficient (PCC) between two tensors.

    There are two different implementations of PCC calculation (estimation):

    - For "small" tensors, the PCC is calculated on the whole (flattened) tensors.

    - For large tensors, we split the tensors into smaller chunks and calculate PCC over each chunk.
      This is done to avoid large memory usage when calculating PCC on large tensors. Unfortunately,
      this isn't the same mathematical operation, but it provides a good estimation of the original approach,
      and is quite simple.

      The calculations over chunks are done in parallel using ThreadPool, to lower the execution time.

      NOTES: original implementation is flattening both tensors and then calculating the PCC, this means
      that the whole tensor is treated as a single random variable with N samples (N = number of elements in the tensor).
      With the estimation procedure we are treating the tensor as a collection of random variables, each chunk is treated
      like a single random variable (so N_chunks in total) with M samples (M = number of elements in the chunk). The final
      PCC is the average of all the PCCs calculated over the chunks.

    """
    # Convert bfloat16 to float32 for PCC calculation - numpy doesn't support bfloat16
    if a.dtype == torch.bfloat16 or b.dtype == torch.bfloat16:
        a = a.type(torch.float32)
        b = b.type(torch.float32)

    a_np = a.detach().numpy().flatten()
    b_np = b.detach().numpy().flatten()
    masked_a = np.ma.masked_invalid(a_np, copy=False)
    masked_b = np.ma.masked_invalid(b_np, copy=False)

    number_of_invalid_elements = np.sum(masked_a.mask)

    # We expect the number of invalid elements to be the same in both tensors
    if number_of_invalid_elements != np.sum(masked_b.mask):
        return 0.0

    # Verify that all invalid elements (nans/infs) are the same in both tensors
    if not np.array_equal(a_np[masked_a.mask], b_np[masked_b.mask], equal_nan=True) or not np.array_equal(
        masked_a.mask, masked_b.mask
    ):
        return 0.0

    # For large tensors, split the tensor into smaller chunks to estimate PCC.
    if a.numel() > tensor_size_threshold:
        pool = ThreadPool()
        results = []

        for i in range(0, masked_a.shape[0], chunk_size):
            work = pool.apply_async(calc_chunk_pcc, args=(masked_a, masked_b, chunk_size, i))
            results.append(work)

        pcc = 0
        n_chunks = len(results)
        for work in results:
            chunk_pcc = work.get()
            pcc += chunk_pcc / n_chunks
    else:
        pcc = np.min(
            np.ma.corrcoef(
                masked_a,
                masked_b,
            )
        )

    if isinstance(pcc, np.ma.core.MaskedConstant):
        return 1.0

    return pcc


def calculate_pcc(a: torch.Tensor, b: torch.Tensor) -> np.float64:
    TENSOR_SIZE_THRESHOLD = int(1e8)
    CHUNK_SIZE = int(1e6)

    return calculate_or_estimate_pcc(a, b, TENSOR_SIZE_THRESHOLD, CHUNK_SIZE)


def compare_pcc(calculated_pcc: float, pcc: float = 0.99):
    assert pcc >= 0, "PCC threshold must be >= 0"
    if calculated_pcc >= pcc:
        logger.trace("PCC is correct")
        return True
    else:
        logger.error("Tensor mismatch. PCC = {}, but required = {}", calculated_pcc, pcc)
        return False


# Deprecated: avoid using it, instead use compare_with_golden
def compare_tensor_to_golden(
    name: str,
    golden: Union[torch.Tensor, tf.Tensor, tf.Variable],
    calculated: torch.Tensor,
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


def calculate_atol(golden, calculated):
    if torch.equal(golden, calculated):
        return 0.0

    if golden.dtype == torch.bool:
        # For bool tensors, return 1.0, since they are not equal (previous check).
        return 1.0

    if golden.dtype == torch.bfloat16:
        # Convert bfloat16 to float32 for PCC calculation - numpy doesn't support bfloat16
        golden = golden.type(torch.float32)
        calculated = calculated.type(torch.float32)

    golden = golden.detach().numpy()
    calculated = calculated.detach().numpy()

    masked_golden = np.ma.masked_invalid(golden, copy=False)
    masked_calculated = np.ma.masked_invalid(calculated, copy=False)

    # Assert that all nans/infs are matching. If not, return inf.
    if not np.array_equal(
        golden[masked_golden.mask], calculated[masked_calculated.mask], equal_nan=True
    ) or not np.array_equal(masked_golden.mask, masked_calculated.mask):
        return torch.inf

    np_max_abs_diff = np.nanmax(np.abs(golden - calculated))
    return torch.tensor(np_max_abs_diff).item()


def calculate_rtol(golden, calculated):
    diff = torch.abs(golden - calculated)
    abs_calculated = torch.abs(calculated)

    # Compute the required rtol (assuming atol=0)
    # Avoid division by zero by setting a high rtol for zero values in b
    safe_abs_calculated = torch.where(abs_calculated == 0, torch.tensor(float("inf")), abs_calculated)

    return torch.max(diff / safe_abs_calculated).item()


def compute_required_tolerances(golden, calculated):
    if golden.shape != calculated.shape:
        raise ValueError("Tensors must have the same shape")

    required_atol = calculate_atol(golden, calculated)
    required_rtol = calculate_rtol(golden, calculated)

    return required_atol, required_rtol


def determine_consistency_limits(
    framework_outputs: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], compiled_outputs: List[torch.Tensor]
) -> Tuple[Optional[float], Optional[float]]:
    """
    Determine the consistency limits between golden (framework) and compiled outputs by computing:
      - The minimum Pearson correlation coefficient (PCC) among non-scalar tensors (lower consistency limit).
      - The maximum absolute tolerance (ATOL) across all outputs (upper deviation limit).

    Parameters:
        framework_outputs: Tuple or list of torch.Tensor representing the expected (golden) outputs.
        compiled_outputs: List of torch.Tensor representing the computed outputs.

    Returns:
        A tuple (min_pcc, max_atol) where:
          - min_pcc is the minimum PCC computed over non-scalar tensors (the lower consistency limit).
            If only scalar outputs are present, this is set to None.
          - max_atol is the maximum absolute tolerance computed over all outputs (the upper deviation limit).
        If the number of framework and compiled outputs do not match, returns (None, None).
    """
    if len(framework_outputs) != len(compiled_outputs):
        logger.error(
            f"Input count mismatch: framework_outputs={len(framework_outputs)}, compiled_outputs={len(compiled_outputs)}"
        )
        return None, None

    pcc_values = []
    atol_values = []

    for idx, (fw_out, co_out) in enumerate(zip(framework_outputs, compiled_outputs), start=1):
        # For non-scalar tensors, ensure the shapes match
        if fw_out.numel() != 1 and fw_out.shape != co_out.shape:
            logger.error(f"Tensor {idx} - Shape mismatch: fw_out shape={fw_out.shape}, co_out shape={co_out.shape}")
            continue

        # If the output is a scalar, compute only ATOL
        if fw_out.numel() == 1:
            atol = calculate_atol(fw_out, co_out)
            atol_values.append(atol)
        else:
            pcc = calculate_pcc(fw_out, co_out)
            pcc_values.append(pcc)
            atol = calculate_atol(fw_out, co_out)
            atol_values.append(atol)

    min_pcc = min(pcc_values) if pcc_values else None
    max_atol = max(atol_values) if atol_values else None

    return min_pcc, max_atol
