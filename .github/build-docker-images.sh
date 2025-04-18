#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

# Compute the hash of the Dockerfile
BRANCH=$(git rev-parse --abbrev-ref HEAD)
DOCKER_TAG=$(./.github/get-docker-tag.sh)
echo "Docker tag: $DOCKER_TAG"

image_name=$1
dockerfile=$2
from_image=$3

GHCR_TOKEN=$(echo $GHCR_TOKEN | base64)
# Check if the image tag already exists in ghcr.io
echo "Checking $REGISTRY with query: https://$REGISTRY/v2/$REPO/$image_name/tags/list"
set +e
match_tag=$(curl -s -H "Authorization: Bearer $GHCR_TOKEN" https://$REGISTRY/v2/$REPO/$image_name/tags/list | jq -r --arg tag "$DOCKER_TAG" '.tags[]' | grep -q "$DOCKER_TAG")
set -e
echo "match_tag: $match_tag"

# Check if the tag exists in ghcr.io
if [ ! -z "$match_tag" ]; then
   echo "Tag $DOCKER_TAG does exist in ghcr.io"
   # Check if the image tag already exists in the local docker
   if docker manifest inspect $REGISTRY/$REPO/$image_name:$DOCKER_TAG > /dev/null; then
       return
   fi
   docker pull $REGISTRY/$REPO/$image_name:$DOCKER_TAG
   return
fi
echo "Tag $DOCKER_TAG does not exist in ghcr.io"

# Prepare build arguments
build_args=("--build-arg" "FROM_TAG=$DOCKER_TAG")
   
# Add FROM_IMAGE tag if declared
if [ -z "$from_image" ]; then
    build_args+=("--build-arg" "FROM_IMAGE=$image_name")
fi
   
# Add main tag if on main branch
if [ "$BRANCH" == "jmcgrath/create-latest-for-all-images" ]; then
    build_args+=("-t" "$REGISTRY/$REPO/$image_name:test-latest")
fi
   
# Add the required tag and dockerfile
build_args+=("-t" "$REGISTRY/$REPO/$image_name:$DOCKER_TAG" "-f" "$dockerfile" ".")
   
# Execute the docker build command with all arguments
echo "Docker build arguments: ${build_args[@]}"
docker build --push "${build_args[@]}"
    