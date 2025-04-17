#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

REPO=ghcr.io/tenstorrent/tt-forge-fe
BASE_IMAGE_NAME=tt-forge-fe-base-ubuntu-22-04
CI_IMAGE_NAME=tt-forge-fe-ci-ubuntu-22-04
BASE_IRD_IMAGE_NAME=tt-forge-fe-base-ird-ubuntu-22-04
IRD_IMAGE_NAME=tt-forge-fe-ird-ubuntu-22-04

# Compute the hash of the Dockerfile
BRANCH=$(git rev-parse --abbrev-ref HEAD)
DOCKER_TAG=$(./.github/get-docker-tag.sh)
echo "Docker tag: $DOCKER_TAG"

build_and_push() {
    local repo=$1
    local image_name=$2
    local dockerfile=$3
    local from_image=$4

    GHCR_TOKEN=$(echo $GHCR_TOKEN | base64)
    curl -H "Authorization: Bearer $GHCR_TOKEN" https://ghcr.io/v2/tenstorrent/$image_name/tags/list

    ## Prepare build arguments
    #build_args=("--build-arg" "FROM_TAG=$DOCKER_TAG")
    #    
    ## Add FROM_IMAGE tag if declared
    #if [ -z "$from_image" ]; then
    #    build_args+=("--build-arg" "FROM_IMAGE=$repo/$image_name")
    #fi
    #    
    ## Add main tag if on specific branch
    #if [ "$BRANCH" == "jmcgrath/create-latest-for-all-images" ]; then
    #    build_args+=("-t" "$repo/$image_name:test-latest")
    #fi
    #    
    ## Add the required tag and dockerfile
    #build_args+=("-t" "$repo/$image_name:$DOCKER_TAG" "-f" "$dockerfile" ".")
    #    
    ## Execute the docker build command with all arguments
    #echo "Docker build arguments: ${build_args[@]}"
    #docker build --push "${build_args[@]}"

    if !docker manifest inspect $repo/$image_name:$DOCKER_TAG > /dev/null; then
        docker pull $repo/$image_name:$DOCKER_TAG
    fi
}

build_and_push $REPO $BASE_IMAGE_NAME .github/Dockerfile.base
build_and_push $REPO $BASE_IRD_IMAGE_NAME .github/Dockerfile.ird base
build_and_push $REPO $CI_IMAGE_NAME .github/Dockerfile.ci
build_and_push $REPO $IRD_IMAGE_NAME .github/Dockerfile.ird ci

echo "All images built and pushed successfully"
echo "CI_IMAGE_NAME:"
echo $CI_IMAGE_NAME:$DOCKER_TAG
