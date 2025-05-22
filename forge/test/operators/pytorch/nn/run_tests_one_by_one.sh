#!/bin/bash

INPUT_FILE="conv_transpose_2d_test_plan_ALL.txt"
PASSED_FILE="passed_ids.txt"
# FAILED_FILE="failed_ids.txt"
FPE_FILE="failed_fpe_ids.txt"
SEGFAULT_FILE="failed_segfault_ids.txt"
KILLED_FILE="failed_killed_ids.txt"
FAILED_FAILED_FILE="failed_failed_ids.txt"
OTHER_FILE="failed_other_ids.txt"

# Čistimo stare fajlove
> "$PASSED_FILE"
# > "$FAILED_FILE"
> "$FPE_FILE"
> "$SEGFAULT_FILE"
> "$KILLED_FILE"
> "$FAILED_FAILED_FILE"
> "$OTHER_FILE"

while IFS= read -r line || [ -n "$line" ]; do
  TEST_ID="$(echo -n "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"

  if [ -z "$TEST_ID" ]; then
    continue
  fi

  echo "TEST_ID=\"$TEST_ID\" pytest"

  env TEST_ID="$TEST_ID" pytest
  EXIT_CODE=$?

  echo "  ↳ Exit code: $EXIT_CODE"

  if [ "$EXIT_CODE" -eq 0 ]; then
    echo "  ↳ Test passed"
    echo "$TEST_ID" >> "$PASSED_FILE"
  else
    echo "  ↳ Test failed (Exit code $EXIT_CODE)"
    # echo "$TEST_ID" >> "$FAILED_FILE"

    case $EXIT_CODE in
      136)
        echo "  ↳ Detected: Floating point exception (SIGFPE)"
        echo "$TEST_ID" >> "$FPE_FILE"
        ;;
      139)
        echo "  ↳ Detected: Segmentation fault (SIGSEGV)"
        echo "$TEST_ID" >> "$SEGFAULT_FILE"
        ;;
      137)
        echo "  ↳ Detected: Killed (SIGKILL)"
        echo "$TEST_ID" >> "$KILLED_FILE"
        ;;
      1)
        echo "  ↳ Detected: Test failed (pytest fail)"
        echo "$TEST_ID" >> "$FAILED_FAILED_FILE"
        ;;
      *)
        echo "  ↳ Detected: Other failure (Exit code $EXIT_CODE)"
        echo "$TEST_ID" >> "$OTHER_FILE"
        ;;
    esac
  fi

  echo
done < "$INPUT_FILE"
