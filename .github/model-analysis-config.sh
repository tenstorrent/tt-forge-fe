#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Declare an associative array to store environment variables
declare -A env_vars

# Model ops test generation
# 1) PR config
env_vars["BRANCH_NAME"]="generate_models_ops_tests"
env_vars["COMMIT_MESSAGE"]="Generate and update models ops tests"
env_vars["TITLE"]="Generate and update models ops tests"
env_vars["BODY"]="This PR will generate models ops tests by extracting the unique ops configurations across all the models present inside the forge/test/models directory path."

# 2) Script config
env_vars["UNIQUE_OPS_OUTPUT_DIR_PATH"]="models_unique_ops_output/"
env_vars["MODELS_OPS_TEST_OUTPUT_DIR_PATH"]="forge/test"
env_vars["MODELS_OPS_TEST_PACKAGE_NAME"]="models_ops"
env_vars["SCRIPT_OUTPUT_LOG"]="generate_models_ops_test.log"
env_vars["GENERATED_MODELS_OPS_TESTS_PATH"]="forge/test/models_ops/"


for key in "${!env_vars[@]}"; do
  echo "$key=${env_vars[$key]}"
done
