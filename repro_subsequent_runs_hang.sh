#!/usr/bin/env bash

set -e

# This script runs the performance benchmark multiple times. In most of the cases, the hang occurs on the second run.

# If the `TT_METAL_FORCE_REINIT` is not set, set it to `1` to prevent the hang on the first run.
if ! [ "$TT_METAL_FORCE_REINIT" ]; then
    echo "Setting workaround environment variable TT_METAL_FORCE_REINIT=1"
    export TT_METAL_FORCE_REINIT=1
fi

if [ $TT_METAL_HOME ]; then
    # `tt-forge-fe` resolves appropriate `TT_METAL_HOME` value automatically. If you have it set, please make sure that it is correct and intentional.
    # In that case remove this check.
    echo "TT_METAL_HOME is set to $TT_METAL_HOME"
    echo "Please make sure that TT_METAL_HOME is set correctly and intentionally - if not, please unset it."
    echo "Exiting..."
    exit 1
fi


for i in {1..24}; do
    echo "Running benchmark iteration $i"
    python forge/test/benchmark/benchmark.py -m resnet50_hf -ts classification -bs 8 -df bfloat16 -lp 32 -o forge-benchmark-e2e-resnet50_hf.json > resnet_$i.log 2>&1
done
