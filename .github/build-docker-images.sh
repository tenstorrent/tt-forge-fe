#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

# postiional arguments
image_name=$1
dockerfile=$2
branch_name=$3
from_image=$4

# Check if the environment variables are set
if [ -z "$DOCKER_TAG" ]; then
    printf "\n$DOCKER_TAG is enviroment var is empty"
    exit 1
fi

if [ -z "$REGISTRY" ]; then
    printf "\n$REGISTRY is enviroment var is empty"
    exit 1
fi

if [ -z "$REPO" ]; then
    printf "\n$REPO is enviroment var is empty"
    exit 1
fi

if [ -z "$GHCR_TOKEN" ]; then
    printf "\n$GHCR_TOKEN is enviroment var is empty"
    exit 1
fi

# Compute the hash of the Dockerfile
BRANCH=$(git rev-parse --abbrev-ref HEAD)
full_image="$REGISTRY/$REPO/$image_name:$DOCKER_TAG"
printf "\n\n## Running build-docker-images.sh for $full_image ##\n\n"

if [ ! -z "$branch_name" ]; then
    printf "\nChecking if image can build on branch $branch_name"
    if [ "$BRANCH" != "$branch_name" ]; then
        printf "\nSkipping build for $full_image since branch $BRANCH does not match $branch_name"
        exit 0
    fi
fi

GHCR_TOKEN=$(printf $GHCR_TOKEN | base64)
# Check if the image tag already exists in ghcr.io
printf "\nChecking $REGISTRY for tag: $DOCKER_TAG"
set +e
match_tag=$(curl -s -H "Authorization: Bearer $GHCR_TOKEN" https://$REGISTRY/v2/$REPO/$image_name/tags/list | jq -r --arg tag "$DOCKER_TAG" '.tags[]' | grep -o $DOCKER_TAG)
set -e

# Check if the tag exists in ghcr.io
if [ ! -z "$match_tag" ]; then
   printf "\n$full_image exists in ghcr.io"

    # Check if the image tag already exists in the local docker
   local_cache=$(docker image ls --filter=reference="$full_image" --format json)

    if [ ! -z "$local_cache" ]; then
        printf "\n$full_image exists in local docker cache"
        if [ "$BRANCH" != "main" ]; then
            printf "\nUsing $full_image for CI Dev branch"
            exit 0
        fi

        printf "\n$full_image updating with latest tag since on main branch"
        docker tag $full_image $REGISTRY/$REPO/$image_name:latest
        docker push $REGISTRY/$REPO/$image_name:latest
        exit 0
    fi
    if [ "$BRANCH" != "main" ]; then
        printf "\nPulling $full_image for CI Dev branch since tag exists in ghcr.io but not in local docker"
        docker pull $full_image
        exit 0
    fi

    printf "\nPulling $full_image and updating with latest tag since on main branch"
    docker pull $full_image
    docker tag $full_image $REGISTRY/$REPO/$image_name:latest
    docker push $REGISTRY/$REPO/$image_name:latest
    exit 0
fi

printf "\n$full_image does not exist in ghcr.io"

build_args=("--build-arg" "FROM_TAG=$DOCKER_TAG")

# Add FROM_IMAGE tag if declared
if [ -z "$from_image" ]; then
    build_args+=("--build-arg" "FROM_IMAGE=$image_name")
fi

# Add the required tag and dockerfile
build_args+=("-t" "$full_image" "-f" "$dockerfile" ".")

# Execute the docker build command with all arguments
printf "\nDocker build arguments: ${build_args[@]}"
docker build --push "${build_args[@]}"
