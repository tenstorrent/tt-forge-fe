#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Installs python requirements for third party tt-mlir project.

# Exit immediately if a command exits with a non-zero status
set -e

source ${TTFORGE_SOURCE_DIR}/env/bin/activate

TT_MLIR_ENV_DIR=${TTFORGE_SOURCE_DIR}/third_party/tt-mlir/env

# Extract LLVM_VERSION from CMakeLists.txt
LLVM_VERSION=$(grep -oP 'set\(LLVM_PROJECT_VERSION "\K[^"]+' ${TT_MLIR_ENV_DIR}/CMakeLists.txt)
echo "LLVM_VERSION: $LLVM_VERSION"
# Set Up Cache Directory for Requirements
REQUIREMENTS_CACHE_DIR=${TTFORGE_SOURCE_DIR}/env/bin/requirements_cache
LLVM_REQUIREMENTS_PATH=${REQUIREMENTS_CACHE_DIR}/llvm-requirements-${LLVM_VERSION}.txt
# Download and Cache LLVM Requirements
if [ ! -e $LLVM_REQUIREMENTS_PATH ]; then
  mkdir -p $REQUIREMENTS_CACHE_DIR
  wget -O $LLVM_REQUIREMENTS_PATH "https://github.com/llvm/llvm-project/raw/$LLVM_VERSION/mlir/python/requirements.txt" --quiet
fi
# Install LLVM Requirements
pip install -r $LLVM_REQUIREMENTS_PATH

# Install tt-mlir requirements
pip install -r ${TT_MLIR_ENV_DIR}/build-requirements.txt