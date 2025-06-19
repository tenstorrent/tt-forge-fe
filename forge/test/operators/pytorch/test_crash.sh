# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Usage:
# source forge/test/operators/pytorch/test_crash.sh

CRASH_SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

# . ${CRASH_SCRIPT_DIR}/test_common.sh

function extract_test_id_from_pytest_collect_function {
    # <Function test_query[no_device-gt-FROM_ANOTHER_OP-None-(1, 4)-None-None]>
    # grep "test_query.py::" | sed -E 's|.*test_query\[no_device-(.*)\]|\1|g'
    grep "<Function test_query" | sed -E 's|.*test_query\[no_device-(.*)\].*|\1|g'
}

function reset_crash_scan {
    # pytest_terminate
    # /opt/tt_metal_infra/scripts/ci/wormhole_b0/reset.sh
    echo "No reset script found, skipping reset."
}

function detect_crashes_scan {

    # local crash_dir="$1"
    local crash_dir="${CRASH_DIR}"

    INPUT_FILE=${crash_dir}/test_ids_crash_scan.txt

    crash_dir=${CRASH_SCRIPT_DIR}/${crash_dir}
    PASSED_FILE="${crash_dir}/passed_ids.txt"
    # FAILED_FILE="failed_ids.txt"
    FPE_FILE="${crash_dir}/failed_fpe_ids.txt"
    SEGFAULT_FILE="${crash_dir}/failed_segfault_ids.txt"
    KILLED_FILE="${crash_dir}/failed_killed_ids.txt"
    FAILED_FILE="${crash_dir}/failed_ids.txt"
    OTHER_FILE="${crash_dir}/failed_other_ids.txt"
    ALL_CRASH_FILE="${crash_dir}/failed_all_crash_ids.txt"

    # Čistimo stare fajlove
    touch "$PASSED_FILE"
    # > "$FAILED_FILE"
    touch "$FPE_FILE"
    touch "$SEGFAULT_FILE"
    touch "$KILLED_FILE"
    touch "$FAILED_FILE"
    touch "$OTHER_FILE"

    # local batch_size=5
    local batch_size=2000
    local start_index=0
    # local start_index=18000
    local end_index=$((start_index + batch_size))
    # local end_index=5
    local total_number=0

    # total_number is number of lines in the input file
    total_number=$(wc -l < "${CRASH_SCRIPT_DIR}/$INPUT_FILE")
    echo "Total number of tests: $total_number"

    # while IFS= read -r line || [ -n "$line" ]; do
    while true; do

      reset_crash_scan

      if [ $start_index -ge $total_number ]; then
        echo "Reached the end of the input file $start_index >= $total_number."


        echo "PASSED_FILE=\"$PASSED_FILE\""
        cat $PASSED_FILE

        echo "FPE_FILE=\"$FPE_FILE\""
        cat $FPE_FILE

        echo "SEGFAULT_FILE=\"$SEGFAULT_FILE\""
        cat $SEGFAULT_FILE

        echo "KILLED_FILE=\"$KILLED_FILE\""
        cat $KILLED_FILE

        echo "FAILED_FILE=\"$FAILED_FILE\""
        cat $FAILED_FILE

        echo "OTHER_FILE=\"$OTHER_FILE\""
        cat $OTHER_FILE

        # echo "# PASSED_FILE=\"$PASSED_FILE\"" >> "$ALL_CRASH_FILE"
        # echo $PASSED_FILE >> "$ALL_CRASH_FILE"
        # echo "# FPE_FILE=\"$FPE_FILE\"" >> "$ALL_CRASH_FILE"
        # echo $FPE_FILE >> "$ALL_CRASH_FILE"
        # echo "# SEGFAULT_FILE=\"$SEGFAULT_FILE\"" >> "$ALL_CRASH_FILE"
        # echo $SEGFAULT_FILE >> "$ALL_CRASH_FILE"
        # echo "# KILLED_FILE=\"$KILLED_FILE\"" >> "$ALL_CRASH_FILE"
        # echo $KILLED_FILE >> "$ALL_CRASH_FILE"
        # echo "# FAILED_FILE=\"$FAILED_FILE\"" >> "$ALL_CRASH_FILE"
        # echo $FAILED_FILE >> "$ALL_CRASH_FILE"
        # echo "# OTHER_FILE=\"$OTHER_FILE\"" >> "$ALL_CRASH_FILE"
        # echo $OTHER_FILE >> "$ALL_CRASH_FILE"

        echo "ALL_CRASH_FILE=\"$ALL_CRASH_FILE\""
        cat $ALL_CRASH_FILE

        break
      fi
      # test_id="$(echo -n "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"

      # if [ -z "$test_id" ]; then
      #   continue
      # fi

      # echo "TEST_ID=\"$test_id\" STATUS_TRACKER=true pytest"

      # env TEST_ID="$test_id" STATUS_TRACKER=true pytest

      local with_params_method=""
      # with_params_method+=" with-log"
      # with_params_method+=" with-params"

      local pytest_params=""
      pytest_params+=" --no-skips"

      local params=""
      params+=" ID_FILES=$INPUT_FILE RANGE=${start_index},${end_index} STATUS_TRACKER=true"

      echo "Params: ${params}"
      echo "Pytest params: ${pytest_params}"

      set +e
      env ${params} ${with_params_method} pytest ${pytest_params}
      EXIT_CODE=$?
      echo "  ↳ Exit code: $EXIT_CODE"
      set -e

      cat test_status.json
      cat test_status.json | jq -r '.test_id'
      cat test_status.json | jq -r '.test_id' > /dev/null 2>&1
      local test_id=$(cat test_status.json | jq -r '.test_id')
      local counter=$(cat test_status.json | jq -r '.counter')
      local status=$(cat test_status.json | jq -r '.status')

      # start_index=start_index+counter+1
      # start_index=$((start_index + counter + 1))
      start_index=$((counter + 1))
      end_index=$((start_index + batch_size))

      # if [ "$EXIT_CODE" -eq 0 ]; then
      if [ "$status" == "completed" ]; then
      # if [ "$status" == "in progress" ]; then

        echo "  ↳ Test passed"
        echo "$test_id" >> "$PASSED_FILE"
        # exit loop

        echo "PASSED_FILE=\"$PASSED_FILE\""
        cat $PASSED_FILE

        echo "FPE_FILE=\"$FPE_FILE\""
        cat $FPE_FILE

        echo "SEGFAULT_FILE=\"$SEGFAULT_FILE\""
        cat $SEGFAULT_FILE

        echo "KILLED_FILE=\"$KILLED_FILE\""
        cat $KILLED_FILE

        echo "FAILED_FILE=\"$FAILED_FILE\""
        cat $FAILED_FILE

        echo "OTHER_FILE=\"$OTHER_FILE\""
        cat $OTHER_FILE

        # break
      else
        echo "  ↳ Test failed (Exit code $EXIT_CODE)"
        echo "$test_id" >> "$ALL_CRASH_FILE"

        # start_index=$((start_index + 1))
        # end_index=$((start_index + batch_size))

        case $EXIT_CODE in
          136)
            echo "  ↳ Detected: Floating point exception (SIGFPE)"
            echo "$test_id" >> "$FPE_FILE"
            ;;
          139)
            echo "  ↳ Detected: Segmentation fault (SIGSEGV)"
            echo "$test_id" >> "$SEGFAULT_FILE"
            ;;
          137)
            echo "  ↳ Detected: Killed (SIGKILL)"
            echo "$test_id" >> "$KILLED_FILE"
            ;;
          1)
            echo "  ↳ Detected: Test failed (pytest fail)"
            echo "$test_id" >> "$FAILED_FILE"
            ;;
          *)
            echo "  ↳ Detected: Other failure (Exit code $EXIT_CODE)"
            echo "$test_id" >> "$OTHER_FILE"
            ;;
        esac
      fi

      echo
    # done < "$INPUT_FILE"
    done
}
