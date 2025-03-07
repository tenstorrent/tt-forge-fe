#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

# Ensure skopeo is installed
if ! command -v skopeo &> /dev/null; then
    echo "skopeo could not be found, installing..."
    sudo apt-get update && sudo apt-get install -y skopeo
fi

REPO=tenstorrent/tt-forge-fe
BASE_IMAGE_NAME=ghcr.io/$REPO/tt-forge-fe-base-ubuntu-22-04
CI_IMAGE_NAME=ghcr.io/$REPO/tt-forge-fe-ci-ubuntu-22-04
BASE_IRD_IMAGE_NAME=ghcr.io/$REPO/tt-forge-fe-base-ird-ubuntu-22-04
IRD_IMAGE_NAME=ghcr.io/$REPO/tt-forge-fe-ird-ubuntu-22-04

# Compute the hash of the Dockerfile
DOCKER_TAG=$(./.github/get-docker-tag.sh)
echo "Docker tag: $DOCKER_TAG"

# Are we on main branch
ON_MAIN=$(git branch --show-current | grep -q main && echo "true" || echo "false")
HAS_CHANGES=$(git diff --quiet origin/main && echo "false" || echo "true")

build_and_push() {
    local image_name=$1
    local dockerfile=$2
    local from_image=$3

    if docker manifest inspect $image_name:$DOCKER_TAG > /dev/null; then
        echo "Image $image_name:$DOCKER_TAG already exists"
    else
        echo "Building image $image_name:$DOCKER_TAG"
        docker build \
            --progress=plain \
            --build-arg FROM_TAG=$DOCKER_TAG \
            ${from_image:+--build-arg FROM_IMAGE=$from_image} \
            -t $image_name:$DOCKER_TAG \
            -f $dockerfile .

        echo "Pushing image $image_name:$DOCKER_TAG"
        docker push $image_name:$DOCKER_TAG
    fi

    # If we are on main branch update manifest and add latest tag
    if [ "$ON_MAIN" = "true" ] && [ "$HAS_CHANGES" = "false" ]; then
        echo "Adding latest tag to image $image_name:$DOCKER_TAG"
        skopeo copy "docker://$image_name:$DOCKER_TAG" "docker://$image_name:latest"
    fi
}

build_and_push $BASE_IMAGE_NAME .github/Dockerfile.base
build_and_push $BASE_IRD_IMAGE_NAME .github/Dockerfile.ird base
build_and_push $CI_IMAGE_NAME .github/Dockerfile.ci
build_and_push $IRD_IMAGE_NAME .github/Dockerfile.ird ci

echo "All images built and pushed successfully"
echo "CI_IMAGE_NAME:"
echo $CI_IMAGE_NAME:$DOCKER_TAG
