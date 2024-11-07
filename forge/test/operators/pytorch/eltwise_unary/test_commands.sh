# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Commands for running unary operators tests

# Run all unary operators tests:
OPERATORS=\
"relu",\
"sqrt",\
"reciprocal",\
"sigmoid",\
"abs",\
"cos",\
"exp",\
"neg",\
"rsqrt",\
"sin",\
"square" \
pytest -svv forge/test/operators/pytorch/test_all.py::test_query -rap
