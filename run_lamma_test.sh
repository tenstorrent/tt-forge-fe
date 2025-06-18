set -o pipefail # Ensures that the exit code reflects the first command that fails
python scripts/model_analysis/run_analysis_and_generate_md_files.py \
    --test_directory_or_file_path "forge/test/models/pytorch/dummy/test_mnist.py"
