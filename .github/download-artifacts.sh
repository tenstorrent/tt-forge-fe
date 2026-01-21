#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Robust artifact downloader for GitHub Actions workflow runs.
# Usage:
#   ./download_artifacts.sh <WORKFLOW_REPO> <RUN_ID> <OUTPUT_DIR> <ARTIFACT_NAME_PREFIX>
# Example:
#   ./download_artifacts.sh tenstorrent/tt-forge-onnx 123456 artifacts "models-unique-ops-config"
set -euxo pipefail

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

# Ensure common utilities present: gh, jq, unzip, curl
install_if_missing() {
  if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y gh jq curl unzip || true
  else
    echo "apt-get not found. Please ensure 'gh', 'jq', 'curl' and 'unzip' are installed in the runner." >&2
  fi
}

if ! command -v gh &>/dev/null || ! command -v jq &>/dev/null || ! command -v curl &>/dev/null || ! command -v unzip &>/dev/null; then
  echo "One or more required tools missing; attempting to install (if apt-get is available)..."
  install_if_missing
fi

echo "GitHub CLI version (if available):"
if command -v gh &>/dev/null; then gh --version | head -n1 || true; fi
echo "jq version (if available):"
if command -v jq &>/dev/null; then jq --version || true; fi

# Authenticate gh CLI non-interactively if token present
echo "Checking GitHub CLI authentication..."
if ! gh auth status &>/dev/null; then
  if [ -n "${GH_TOKEN:-}" ]; then
    echo "Authenticating gh using GH_TOKEN..."
    printf '%s' "$GH_TOKEN" | gh auth login --with-token || true
  elif [ -n "${GITHUB_TOKEN:-}" ]; then
    echo "Authenticating gh using GITHUB_TOKEN..."
    printf '%s' "$GITHUB_TOKEN" | gh auth login --with-token || true
  else
    echo "GitHub CLI is not authenticated and no GH_TOKEN/GITHUB_TOKEN present. Please set GH_TOKEN or GITHUB_TOKEN." >&2
    exit 1
  fi

  if ! gh auth status &>/dev/null; then
    echo "Failed to authenticate gh CLI after attempting token login." >&2
    gh auth status || true
    exit 1
  fi
else
  gh auth status || true
fi

# Prepare output directory: remove contents (safe guard)
echo "Preparing output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
rm -rf "${OUTPUT_DIR:?}"/*

# Helper: merge paginated artifacts into a single JSON object {artifacts: [...]}
fetch_artifacts_json() {
  # Use jq -s to join multiple JSON docs safely; ensure we get an artifacts array even if empty.
  gh api --paginate "/repos/$WORKFLOW_REPO/actions/runs/$RUN_ID/artifacts" 2>/dev/null \
    | jq -s 'map(.artifacts) | add // [] | {artifacts: .}'
}

ARTIFACTS_JSON="$(fetch_artifacts_json || true)"

if [ -z "$ARTIFACTS_JSON" ] || [ "$ARTIFACTS_JSON" = "null" ]; then
  echo "Failed to fetch artifacts or empty response for run ID $RUN_ID" >&2
  exit 1
fi

# Validate artifact count is a single integer
ART_COUNT="$(jq -r '.artifacts | length // 0' <<<"$ARTIFACTS_JSON")"
if ! printf '%s' "$ART_COUNT" | grep -qE '^[0-9]+$'; then
  echo "Unexpected artifact count value: $ART_COUNT" >&2
  echo "Full artifacts payload:" >&2
  jq -r '.' <<<"$ARTIFACTS_JSON" >&2 || true
  exit 1
fi

if [ "$ART_COUNT" -eq 0 ]; then
  echo "No artifacts found for run ID $RUN_ID"
  jq -r '.' <<<"$ARTIFACTS_JSON" || true
  exit 1
fi

# Build array of matching artifact names using the prefix (safe: exact prefix match)
artifact_names=()
while IFS= read -r name; do
  artifact_names+=("$name")
done < <(jq -r --arg prefix "$PREFIX" '.artifacts[] | select(.name | startswith($prefix)) | .name' <<<"$ARTIFACTS_JSON")

if [ "${#artifact_names[@]}" -eq 0 ]; then
  echo "No artifacts matching prefix '$PREFIX' found for run $RUN_ID"
  echo "Available artifacts:"
  jq -r '.artifacts[] | .name' <<<"$ARTIFACTS_JSON" || true
  exit 1
fi

# Download with retries wrapper (attempt gh run download first)
download_with_retry() {
  local run_id="$1"
  local repo="$2"
  local name="$3"
  local dir="$4"
  local tries=0
  local max_tries=5
  local sleep_seconds=2

  while true; do
    set +e
    gh run download "$run_id" --repo "$repo" --name "$name" --dir "$dir"
    rc=$?
    set -e
    if [ $rc -eq 0 ]; then
      return 0
    fi

    tries=$((tries + 1))
    echo "Warning: 'gh run download' failed for '$name' (attempt $tries/$max_tries), rc=$rc"
    if [ $tries -ge $max_tries ]; then
      echo "Giving up on 'gh run download' for '$name' after $max_tries attempts."
      return 1
    fi
    sleep $sleep_seconds
    sleep_seconds=$((sleep_seconds * 2))
  done
}

# Fallback: download by artifact id using REST API (zip endpoint) and unzip it
download_by_id_fallback() {
  local repo="$1"
  local run_id="$2"
  local name="$3"
  local outdir="$4"

  echo "Attempting REST API fallback for artifact '$name'..."

  # Re-fetch merged artifacts JSON to ensure freshness
  local art_json
  art_json="$(fetch_artifacts_json || true)"
  if [ -z "$art_json" ] || [ "$art_json" = "null" ]; then
    echo "Failed to fetch artifacts JSON for fallback" >&2
    return 2
  fi

  # Try to find an exact match first, then a prefix match
  local art_id
  art_id="$(jq -r --arg nm "$name" '.artifacts[] | select(.name == $nm) | .id' <<<"$art_json" | head -n1 || true)"
  if [ -z "$art_id" ] || [ "$art_id" = "null" ]; then
    art_id="$(jq -r --arg nm "$name" '.artifacts[] | select(.name | startswith($nm)) | .id' <<<"$art_json" | head -n1 || true)"
  fi

  if [ -z "$art_id" ] || [ "$art_id" = "null" ]; then
    echo "Could not find artifact id for name '$name' in run $run_id" >&2
    # Help debugging: print available names that start with prefix
    jq -r --arg nm "$name" '.artifacts[] | .name' <<<"$art_json" || true
    return 3
  fi

  # Determine token for Authorization
  local token
  if [ -n "${GH_TOKEN:-}" ]; then
    token="$GH_TOKEN"
  elif [ -n "${GITHUB_TOKEN:-}" ]; then
    token="$GITHUB_TOKEN"
  else
    echo "No GH_TOKEN or GITHUB_TOKEN available for REST API download" >&2
    return 4
  fi

  mkdir -p "$outdir"
  local tmpzip
  tmpzip="$(mktemp --suffix=.zip)"
  # Use the artifact id endpoint (redirects to CDN)
  set +e
  curl -sSL -H "Authorization: token $token" -o "$tmpzip" "https://api.github.com/repos/$repo/actions/artifacts/$art_id/zip"
  rc=$?
  set -e
  if [ $rc -ne 0 ] || [ ! -s "$tmpzip" ]; then
    echo "Failed to download artifact zip for id $art_id (rc=$rc)" >&2
    rm -f "$tmpzip"
    return 5
  fi

  unzip -q "$tmpzip" -d "$outdir"
  rc=$?
  rm -f "$tmpzip"
  if [ $rc -ne 0 ]; then
    echo "Failed to unzip artifact $tmpzip into $outdir (rc=$rc)" >&2
    return 6
  fi

  return 0
}

# Main download loop: try gh run download (with retry), fallback to REST API if needed
echo "Downloading ${#artifact_names[@]} selected artifact(s)..."
for name in "${artifact_names[@]}"; do
  echo "Downloading artifact: $name"
  mkdir -p "$OUTPUT_DIR/$name"

  if download_with_retry "$RUN_ID" "$WORKFLOW_REPO" "$name" "$OUTPUT_DIR/$name"; then
    echo "Downloaded '$name' via gh run download."
    continue
  fi

  echo "gh run download failed repeatedly for '$name' — trying REST API fallback..."
  if download_by_id_fallback "$WORKFLOW_REPO" "$RUN_ID" "$name" "$OUTPUT_DIR/$name"; then
    echo "Downloaded '$name' via REST API fallback."
    continue
  fi

  echo "ERROR: Failed to download artifact '$name' by both gh run download and REST API fallback." >&2
  exit 1
done

echo "Done: all matching artifacts downloaded into $OUTPUT_DIR"
