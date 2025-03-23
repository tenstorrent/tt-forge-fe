#!/bin/bash
set -e

# Define the base extraction directory
BASE_EXTRACTED_DIR="test_log_artifacts"

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo "GitHub CLI is not installed. Please install it first:"
    echo "https://cli.github.com/manual/installation"
    exit 1
fi

# Check if authenticated with GitHub
if ! gh auth status &> /dev/null; then
    echo "You need to authenticate with GitHub first. Run: gh auth login"
    exit 1
fi

# Set the repository and workflow filename
REPO="tenstorrent/tt-forge-fe"
WORKFLOW_FILE="on-nightly-models-ops.yml"

# Default artifact filters if none provided
DEFAULT_FILTERS=(
    "test-log-n150-1"
    "test-log-n150-2"
    "test-log-n150-3"
    "test-log-n150-4"
)

# Function to display help
show_help() {
    echo "Usage: $0 [--run-id <workflow-run-id>] [--all] [filter1] [filter2] ..."
    echo ""
    echo "Downloads artifacts from a GitHub Actions workflow run for ${WORKFLOW_FILE} in ${REPO}."
    echo ""
    echo "Arguments:"
    echo "  --run-id <workflow-run-id>  Optional workflow run ID to download artifacts from."
    echo "                              If not provided, the script will fetch the latest finished run."
    echo "  --all                       Download all artifacts with no filtering."
    echo "  [filter1] [filter2] ...     Optional artifact name filters. If provided, only"
    echo "                              artifacts matching these filters will be downloaded."
    echo "                              If none provided (and --all not used), the following defaults will be used:"
    echo "                              ${DEFAULT_FILTERS[*]}"
    echo ""
    echo "Examples:"
    echo "  $0                                     # Download default artifacts from the latest finished run"
    echo "  $0 --run-id 123456789                  # Download default artifacts from run ID 123456789"
    echo "  $0 --run-id 123456789 test-log-n150-1  # Download only test-log-n150-1 from run ID 123456789"
    echo "  $0 --all test-log                      # Download all artifacts matching 'test-log' from the latest finished run"
    echo ""
}

# Show help if requested
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    show_help
    exit 0
fi

# Process optional --run-id argument
WORKFLOW_RUN_ID=""
if [ "$1" == "--run-id" ]; then
    shift
    if [ -z "$1" ]; then
        echo "Error: --run-id requires a workflow run ID." >&2
        exit 1
    fi
    WORKFLOW_RUN_ID="$1"
    shift
fi

# Process optional --all flag for artifacts filtering
if [ "$1" == "--all" ]; then
    ARTIFACT_FILTERS=("")  # Empty filter matches everything
    shift
else
    # If artifact filters are provided, use them; otherwise, use defaults.
    if [ $# -gt 0 ]; then
        ARTIFACT_FILTERS=("$@")
    else
        ARTIFACT_FILTERS=("${DEFAULT_FILTERS[@]}")
    fi
fi

# Function to get the latest finished workflow run ID
get_latest_run_id() {
    echo "Finding latest finished run for workflow: ${WORKFLOW_FILE} in ${REPO}..." >&2

    # Query for runs that have completed
    LATEST_RUN=$(gh api "/repos/${REPO}/actions/workflows/${WORKFLOW_FILE}/runs?status=completed&per_page=1" --jq '.workflow_runs[0]')

    if [ -z "$LATEST_RUN" ] || [ "$LATEST_RUN" == "null" ]; then
        echo "No finished runs found for workflow: ${WORKFLOW_FILE}" >&2
        exit 1
    fi

    RUN_ID=$(echo "$LATEST_RUN" | jq -r '.id')
    RUN_NAME=$(echo "$LATEST_RUN" | jq -r '.name')
    RUN_STATUS=$(echo "$LATEST_RUN" | jq -r '.status')
    RUN_CONCLUSION=$(echo "$LATEST_RUN" | jq -r '.conclusion')
    RUN_CREATED=$(echo "$LATEST_RUN" | jq -r '.created_at')

    echo "Latest finished run found:" >&2
    echo "  ID: ${RUN_ID}" >&2
    echo "  Name: ${RUN_NAME}" >&2
    echo "  Status: ${RUN_STATUS}" >&2
    echo "  Conclusion: ${RUN_CONCLUSION}" >&2
    echo "  Created at: ${RUN_CREATED}" >&2

    echo "$RUN_ID"
}

FOUND_MATCHES=0

# Function to download artifacts and rename log files
download_artifacts() {
    local run_id=$1

    echo "Listing artifacts for run ID ${run_id} from ${REPO}..."

    ARTIFACTS_JSON=$(gh api "/repos/${REPO}/actions/runs/${run_id}/artifacts" --jq '.')
    ARTIFACTS_COUNT=$(echo "$ARTIFACTS_JSON" | jq '.total_count')
    if [ "$ARTIFACTS_COUNT" -eq 0 ]; then
        echo "No artifacts found for this run."
        return
    fi

    echo "$ARTIFACTS_JSON" | jq -r '.artifacts[] | "\(.id) \(.name) \(.size_in_bytes)"' | while read -r ID NAME SIZE_BYTES; do
        # Check if the artifact name matches any of our filters
        for FILTER in "${ARTIFACT_FILTERS[@]}"; do
            if [[ "$NAME" == *"$FILTER"* ]]; then
                FOUND_MATCHES=1
                echo "Downloading artifact: $NAME (matches filter: $FILTER)"

                # Create a dedicated subdirectory for this artifact
                ARTIFACT_DIR="${BASE_EXTRACTED_DIR}/${NAME}"
                mkdir -p "$ARTIFACT_DIR"

                # Download and extract the artifact into its subdirectory
                gh run download "$run_id" -n "$NAME" -D "$ARTIFACT_DIR" -R "$REPO"
                if [ $? -eq 0 ]; then
                    echo "Successfully downloaded artifact: ${NAME}"

                    # Find the first log file in the artifact directory
                    LOG_FILE=$(find "$ARTIFACT_DIR" -maxdepth 1 -name "*.log" -type f | head -n 1)
                    if [ -n "$LOG_FILE" ]; then
                        NEW_NAME="${BASE_EXTRACTED_DIR}/${NAME}.log"
                        mv "$LOG_FILE" "$NEW_NAME"
                    else
                        echo "No log file found in artifact: $NAME"
                    fi
                else
                    echo "Failed to download artifact: ${NAME}"
                fi
                break
            fi
        done
    done

    # Clean up any remaining zip files
    find "$BASE_EXTRACTED_DIR" -name "*.zip" -type f -delete

    # Remove all artifact subdirectories, leaving only the renamed log files
    find "$BASE_EXTRACTED_DIR" -mindepth 1 -type d -exec rm -rf {} +

    if [ "$FOUND_MATCHES" -eq 0 ]; then
        echo "No artifacts matched the specified filters."
        echo "Available filters were: ${ARTIFACT_FILTERS[*]}"
        echo "Run with different filter arguments if you want to download other artifacts."
    else
        echo "All matching artifacts have been processed."
        echo "Final log files in ${BASE_EXTRACTED_DIR}:"
        ls -la "$BASE_EXTRACTED_DIR"
    fi
}

# Clean up any existing artifacts directory
if [ -d "$BASE_EXTRACTED_DIR" ]; then
    echo "Cleaning up existing ${BASE_EXTRACTED_DIR} directory..."
    rm -rf "$BASE_EXTRACTED_DIR"
fi
mkdir -p "$BASE_EXTRACTED_DIR"

# If a workflow run ID was not provided, fetch the latest finished run ID
if [ -z "$WORKFLOW_RUN_ID" ]; then
    WORKFLOW_RUN_ID=$(get_latest_run_id)
fi

download_artifacts "$WORKFLOW_RUN_ID"

echo "Download complete! All requested artifacts have been processed."
