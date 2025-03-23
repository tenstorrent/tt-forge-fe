BASE_EXTRACTED_DIR="test_log_artifacts"

# Run download_nightly_logs.sh to download the log files.
./download_nightly_logs.sh

# Run the model analysis script using the downloaded log files.
# The log files are expected to be in the BASE_EXTRACTED_DIR directory.
echo "Running update..."
python scripts/model_analysis/models_ops_test_failure_update.py \
    --log_files "${BASE_EXTRACTED_DIR}/test-log-n150-1.log" \
                "${BASE_EXTRACTED_DIR}/test-log-n150-2.log" \
                "${BASE_EXTRACTED_DIR}/test-log-n150-3.log" \
                "${BASE_EXTRACTED_DIR}/test-log-n150-4.log" \
    --use_report

echo "Update completed!"
