# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import yaml


def get_execution_flow():
    # Load YAML from a file
    with open("execution_flow.yaml", "r") as file:
        execution_flow = yaml.safe_load(file)  # Use safe_load to avoid security risks

    def flatten_data(data):
        flattened_data = []
        for k, v in data.items():
            if isinstance(v, dict):
                flattened_data.extend(flatten_data(v))
            else:
                flattened_data.append({k: v})
        return flattened_data

    flattened_exec_flow = flatten_data(execution_flow)

    return flattened_exec_flow


def get_execution_depth(tb):
    exec_flow = get_execution_flow()
    traceable_funcs = list(map(lambda flow: list(flow.keys())[0], exec_flow))

    exec_depth = "UNKNOWN::UNKNOWN"
    for trace in tb[::-1]:
        if trace.name in traceable_funcs:
            exec_depth = list(exec_flow[traceable_funcs.index(trace.name) - 1].values())[0]
    return exec_depth
