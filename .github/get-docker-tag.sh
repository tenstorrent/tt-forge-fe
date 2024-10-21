#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Calculate hash for docker image tag.
# The hash is based on the MLIR docker tag  and the hash of the Dockerfile(s).

if [ -f third_party/tt-mlir/.github/get-docker-tag.sh ]; then
    MLIR_DOCKER_TAG=$(cd third_party/tt-mlir && .github/get-docker-tag.sh)
else
    MLIR_DOCKER_TAG="default-tag"
fi
DOCKERFILE_HASH_FILES=".github/Dockerfile.base .github/Dockerfile.ci"
DOCKERFILE_HASH=$( (echo $MLIR_DOCKER_TAG; sha256sum $DOCKERFILE_HASH_FILES) | sha256sum | cut -d ' ' -f 1)
echo dt-$DOCKERFILE_HASH
