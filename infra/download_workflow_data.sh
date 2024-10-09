# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

#!/bin/bash

# Ensure required tools are installed
for tool in gh jq; do
    if ! command -v $tool &> /dev/null; then
        echo "$tool could not be found. Please install it to proceed."
        exit 1
    fi
done

# Ensure required inputs are provided
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: $0 <repository> <run-id> <attempt-number>"
    exit 1
fi

REPOSITORY=$1
RUN_ID=$2
ATTEMPT_NUMBER=$3

set_up_dirs() {
    local workflow_run_id=$1
    mkdir -p generated/cicd
    rm -rf generated/cicd/$workflow_run_id
    mkdir -p generated/cicd/$workflow_run_id/artifacts
    mkdir -p generated/cicd/$workflow_run_id/logs
}

download_artifacts() {
    if gh api --paginate /repos/$REPOSITORY/actions/runs/$RUN_ID/artifacts | jq '.artifacts[] | .name' | grep -q "test-reports-*"; then
        echo "[Info] Downloading artifacts $attempt_number"
        gh run download --repo $REPOSITORY -D generated/cicd/$RUN_ID/artifacts --pattern "test-reports-*" $RUN_ID
    else
        echo "[Warning] Test reports not found for workflow run $RUN_ID"
    fi
}

download_logs_for_all_jobs() {
    local max_attempts=$3

    echo "[info] Downloading logs for job with id $job_id for all attempts up to $max_attempts"
    for attempt_number in $(seq 1 $max_attempts); do
        echo "[Info] Downloading for attempt $attempt_number"

        gh api /repos/$REPOSITORY/actions/runs/$RUN_ID/attempts/$attempt_number/jobs --paginate | jq '.jobs[].id' | while read -r job_id; do
            echo "[info] Download logs for job with id $job_id, attempt number $attempt_number"
            gh api /repos/$REPOSITORY/actions/jobs/$job_id/logs > generated/cicd/$RUN_ID/logs/$job_id.log
        done
    done
}

set_up_dirs "$RUN_ID"
download_artifacts "$REPOSITORY" "$RUN_ID"
download_logs_for_all_jobs "$REPOSITORY" "$RUN_ID" "$ATTEMPT_NUMBER"
gh api /repos/$REPOSITORY/actions/runs/$RUN_ID/attempts/$ATTEMPT_NUMBER > generated/cicd/$RUN_ID/workflow.json
gh api /repos/$REPOSITORY/actions/runs/$RUN_ID/attempts/$ATTEMPT_NUMBER/jobs --paginate  | jq -s '{total_count: .[0].total_count, jobs: map(.jobs) | add}' > generated/cicd/$RUN_ID/workflow_jobs.json

find generated/cicd/$RUN_ID -type f -exec ls -lh {} \;
