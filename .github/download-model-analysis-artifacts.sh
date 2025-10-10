#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -euxo pipefail

# Validate required positional arguments
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <WORKFLOW_REPO> <RUN_ID> <OUTPUT_DIR>"
  exit 1
fi

WORKFLOW_REPO="$1"
RUN_ID="$2"
OUTPUT_DIR="$3"

echo "Installing prerequisites (gh, jq)..."
if ! command -v gh &> /dev/null; then
  sudo apt update -qq
  sudo apt install -y gh jq
else
  echo "gh already installed"
fi

echo "Checking GitHub CLI authentication..."
if ! gh auth status &> /dev/null; then
  echo "GitHub CLI is not authenticated. Please run 'gh auth login' or set GH_TOKEN."
  exit 1
else
  gh auth status
fi

# Clean the output directory
echo "Cleaning output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
rm -rf "$OUTPUT_DIR"/*

# Get list of artifact names
echo "Fetching artifacts from run ID: $RUN_ID in $WORKFLOW_REPO"
ARTIFACTS_JSON="$(gh api --paginate "/repos/$WORKFLOW_REPO/actions/runs/$RUN_ID/artifacts" --jq '.artifacts // []')"

if [ "$(jq 'length' <<<"$ARTIFACTS_JSON")" -eq 0 ]; then
  echo "No artifacts found for run ID $RUN_ID"
  exit 1
fi

# Filter artifact names starting with "models-unique-ops-config"
artifact_names=()
while IFS= read -r name; do
  artifact_names+=("$name")
done < <(jq -r '.[] | select(.name | startswith("models-unique-ops-config")) | .name' <<< "$ARTIFACTS_JSON")

if [ "${#artifact_names[@]}" -eq 0 ]; then
  echo "No matching artifacts found"
  exit 1
fi

# Download matching artifacts using gh run download
echo "Downloading selected artifacts using gh run download..."
for name in "${artifact_names[@]}"; do
  echo "Downloading artifact: $name"
  mkdir -p "$OUTPUT_DIR"/"$name"
  gh run download "$RUN_ID" \
    --repo "$WORKFLOW_REPO" \
    --name "$name" \
    --dir "$OUTPUT_DIR"/"$name"
done

echo "Done: all matching artifacts downloaded in $OUTPUT_DIR"
