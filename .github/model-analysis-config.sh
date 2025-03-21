#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# If set to true, it will set the environment variables for models ops test generation otherwise markdown generation env variables will be set
GENERATE_MODELS_OPS_TEST=$1

# Declare an associative array to store environment variables
declare -A env_vars

# Markdown Generation
# 1) PR config
env_vars["BRANCH_NAME"]="model_analysis"
env_vars["COMMIT_MESSAGE"]="Update model analysis documentation"
env_vars["TITLE"]="Update model analysis documentation"
env_vars["BODY"]="This PR will update model analysis documentation."
env_vars["OUTPUT_PATH"]="model_analysis_docs/"

# 2) Script config
env_vars["MARDOWN_DIR_PATH"]="./model_analysis_docs"
env_vars["SCRIPT_OUTPUT_LOG"]="model_analysis.log"


# Model ops test generation
# 1) Script config
env_vars["MODELS_OPS_TEST_OUTPUT_DIR_PATH"]="forge/test"
env_vars["MODELS_OPS_TEST_PACKAGE_NAME"]="models_ops"


# Common Config for markdown generation and model ops test generation
env_vars["TEST_DIR_OR_FILE_PATH"]="forge/test/models/pytorch"
env_vars["UNIQUE_OPS_OUTPUT_DIR_PATH"]="./models_unique_ops_output"


# If GENERATE_MODELS_OPS_TEST is set to true, Modify the PR config to model ops test generation.
if [[ "$GENERATE_MODELS_OPS_TEST" == "true" ]]; then
    env_vars["BRANCH_NAME"]="generate_models_ops_test"
    env_vars["COMMIT_MESSAGE"]="Generate and update models ops tests"
    env_vars["TITLE"]="Generate and update models ops tests"
    env_vars["BODY"]="This PR will generate models ops tests by extracting the unique ops configurations across all the pytorch models present inside the forge/test/models directory path."
    env_vars["OUTPUT_PATH"]="forge/test/models_ops/"
    env_vars["SCRIPT_OUTPUT_LOG"]="generate_models_ops_test.log"
fi


for key in "${!env_vars[@]}"; do
  echo "$key=${env_vars[$key]}"
done
