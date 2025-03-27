#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# parameters:
# <mlir_file> <json_file>

echo "run ttmlir-opt on $1"
./install/bin/ttmlir-opt --tt-register-device="system-desc-path=ttrt-artifacts/system_desc.ttsys" --ttir-to-ttnn-backend-pipeline $1 -o ttnn.mlir
echo "run ttmlir-translate"
./install/bin/ttmlir-translate --ttnn-to-flatbuffer ttnn.mlir -o out.ttnn
echo "run ttrt-perf"
ttrt perf out.ttnn
echo "run device_perf.py creating $2"
python ./forge/test/benchmark/device_perf.py -cdp ttrt-artifacts/out.ttnn/perf/ops_perf_results.csv $2
