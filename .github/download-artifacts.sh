#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -euxo pipefail

# Usage:
#   ./download_artifacts.sh <WORKFLOW_REPO> <RUN_ID> <OUTPUT_DIR> <ARTIFACT_NAME_PREFIX>
# Example:
#   ./download_artifacts.sh tenstorrent/tt-forge-fe 123456 artifacts "models-unique-ops-config"

# Validate positional arguments (exactly 4 required)
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <WORKFLOW_REPO> <RUN_ID> <OUTPUT_DIR> <ARTIFACT_NAME_PREFIX>"
  exit 1
fi

WORKFLOW_REPO="$1"
RUN_ID="$2"
OUTPUT_DIR="$3"
PREFIX="$4"

echo "Repository: $WORKFLOW_REPO"
echo "Run ID: $RUN_ID"
echo "Output directory: $OUTPUT_DIR"
echo "Artifact name prefix: $PREFIX"

echo "Installing prerequisites (gh, jq) if missing..."
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

# Clean (create) the output directory
echo "Preparing output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
rm -rf "$OUTPUT_DIR"/*

# Fetch the artifacts list JSON for the run
echo "Fetching artifacts for run ID: $RUN_ID in $WORKFLOW_REPO"
ARTIFACTS_JSON="$(gh api --paginate "/repos/$WORKFLOW_REPO/actions/runs/$RUN_ID/artifacts" --raw 2>/dev/null || true)"

# Validate JSON
if [ -z "$ARTIFACTS_JSON" ]; then
  echo "Failed to fetch artifacts or empty response for run ID $RUN_ID"
  exit 1
fi

# Check for any artifacts
ART_COUNT="$(jq '.artifacts | length' <<<"$ARTIFACTS_JSON" 2>/dev/null || echo "0")"
if [ "$ART_COUNT" -eq 0 ]; then
  echo "No artifacts found for run ID $RUN_ID"
  exit 1
fi

# Filter artifact names starting with the requested prefix (use jq --arg for safety)
artifact_names=()
while IFS= read -r name; do
  artifact_names+=("$name")
done < <(
  jq -r --arg prefix "$PREFIX" '.artifacts[] | select(.name | startswith($prefix)) | .name' <<<"$ARTIFACTS_JSON"
)

if [ "${#artifact_names[@]}" -eq 0 ]; then
  echo "No artifacts matching prefix '$PREFIX' found for run $RUN_ID"
  echo "Available artifacts:"
  jq -r '.artifacts[] | .name' <<<"$ARTIFACTS_JSON" || true
  exit 1
fi

# Download matching artifacts using gh run download
echo "Downloading ${#artifact_names[@]} selected artifact(s) using 'gh run download'..."
for name in "${artifact_names[@]}"; do
  echo "Downloading artifact: $name"
  mkdir -p "$OUTPUT_DIR/$name"
  gh run download "$RUN_ID" \
    --repo "$WORKFLOW_REPO" \
    --name "$name" \
    --dir "$OUTPUT_DIR/$name"
done

echo "Done: all matching artifacts downloaded into $OUTPUT_DIR"
