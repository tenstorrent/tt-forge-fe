# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Cleans all build artifacts, except for the environment and toolchains.
# If you want to clean the environment (should be rarely needed), remove the env/build directories.
# E.g. rm -rf env/build third_party/tt-mlir/env/build

rm -rf build
rm -rf third_party/tt-mlir/build
rm -rf third_party/tt-mlir/third_party/tt-metal
rm -rf third_party/tvm/build
