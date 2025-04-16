#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

REPO=tenstorrent/tt-forge-fe
BASE_IMAGE_NAME=ghcr.io/$REPO/tt-forge-fe-base-ubuntu-22-04
CI_IMAGE_NAME=ghcr.io/$REPO/tt-forge-fe-ci-ubuntu-22-04
BASE_IRD_IMAGE_NAME=ghcr.io/$REPO/tt-forge-fe-base-ird-ubuntu-22-04
IRD_IMAGE_NAME=ghcr.io/$REPO/tt-forge-fe-ird-ubuntu-22-04

# Compute the hash of the Dockerfile
BRANCH=$(git rev-parse --abbrev-ref HEAD)
DOCKER_TAG=$(./.github/get-docker-tag.sh)
echo "Docker tag: $DOCKER_TAG"

build_and_push() {
    local image_name=$1
    local dockerfile=$2
    local from_image=$3
a
    GHCR_TOKEN=$(echo $GHCR_TOKEN | base64)
    curl -H "Authorization: Bearer $GHCR_TOKEN" https://ghcr.io/v2/tenstorrent/$image_name/tags/list

    # Prepare build arguments
    build_args=("--build-arg" "FROM_TAG=$DOCKER_TAG")
        
    # Add FROM_IMAGE tag if declared
    if [ -z "$from_image" ]; then
        build_args+=("--build-arg" "FROM_IMAGE=$image_name")
    fi
        
    # Add main tag if on specific branch
    if [ "$BRANCH" == "jmcgrath/create-latest-for-all-images" ]; then
        build_args+=("-t" "$image_name:test-latest")
    fi
        
    # Add the required tag and dockerfile
    build_args+=("-t" "$image_name:$DOCKER_TAG" "-f" "$dockerfile" ".")
        
    # Execute the docker build command with all arguments
    echo "Docker build arguments: ${build_args[@]}"
    docker build --push "${build_args[@]}"
}

build_and_push $BASE_IMAGE_NAME .github/Dockerfile.base
build_and_push $BASE_IRD_IMAGE_NAME .github/Dockerfile.ird base
build_and_push $CI_IMAGE_NAME .github/Dockerfile.ci
build_and_push $IRD_IMAGE_NAME .github/Dockerfile.ird ci

echo "All images built and pushed successfully"
echo "CI_IMAGE_NAME:"
echo $CI_IMAGE_NAME:$DOCKER_TAG
