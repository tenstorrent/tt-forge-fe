#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# parameters:
# <mlir_file> <json_file> <ttnn_mlir_output>

# Check all arguments provided
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Error: Missing arguments. Usage: compile_and_run.sh <mlir_file> <json_file> <ttnn_mlir_output>"
    exit 1
fi

echo "run ttmlir-opt on $1"
./install/bin/ttmlir-opt --tt-register-device="system-desc-path=ttrt-artifacts/system_desc.ttsys" --ttir-to-ttnn-backend-pipeline $1 -o $3
if [ $? -ne 0 ]; then
    echo "Error: TTmlir opt command failed."
    exit 1
fi
echo "run ttmlir-translate"
./install/bin/ttmlir-translate --ttnn-to-flatbuffer $3 -o out.ttnn
if [ $? -ne 0 ]; then
    echo "Error: TTmlir translate command failed."
    exit 1
fi
echo "run ttrt-perf"
ttrt perf out.ttnn
if [ $? -ne 0 ]; then
    echo "Error: TTRT perf command failed."
    exit 1
fi
echo "run device_perf.py creating $2"
python ./forge/test/benchmark/device_perf.py -cdp ttrt-artifacts/out.ttnn/perf/ops_perf_results.csv $2
csv_file="${2%.*}.csv"
cp ttrt-artifacts/out.ttnn/perf/ops_perf_results.csv "$csv_file"
if [ $? -ne 0 ]; then
    echo "Error: Failed to copy ops_perf_results.csv to $csv_file."
    exit 1
fi
echo "Copied ops_perf_results.csv to $csv_file"
