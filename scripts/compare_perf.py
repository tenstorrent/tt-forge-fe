# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import math
import argparse
from elasticsearch import Elasticsearch
import pandas as pd

# add project root to search path
project_root_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(project_root_path)

from third_party.confidential_keys.elastic_search import ES_ENDPOINT, ES_USERNAME, ES_PASSWORD


def get_perf_from_es(es, build_id):
    """ Get the perf from elastic search for the specified build id. """
    query = {
        "query": {
            "term": {
                "build_id.keyword": build_id
            }
        },
        "size": 100
    }
    perf_res = es.search(index="pybuda-perf-ci", body=query)
    hits = perf_res['hits']['hits']
    return [{
        "build_id": h["_source"]["build_id"],
        "model": h["_source"]["args"]['model'],
        "config": h["_source"]["args"]['config'],
        "dataformat": h["_source"]["args"]['dataformat'],
        "arch": h["_source"]["args"]['arch'],
        "samples_per_sec": h["_source"]["samples_per_sec"],
        "index": "_".join([h["_source"]["args"]['model'], h["_source"]["args"]['config'], h["_source"]["args"]['arch'], h["_source"]["args"]['dataformat']])
    } for h in hits]


def compare_perf(build_ids: list):
    """ Compare the perf for the specified build ids. """
    es = Elasticsearch([ES_ENDPOINT], http_auth=(ES_USERNAME, ES_PASSWORD))
    print("\nGetting perf from elastic search\n")

    # Load perf data from elastic search
    data = []
    for build_id in build_ids:
        es_res = get_perf_from_es(es, build_id)
        data.extend(es_res)
        print(f"Got perf for build id: {build_id} with {len(es_res)} records")

    df = pd.DataFrame.from_records(data)
    ids = df['build_id'].unique()

    # pivot table on build_id
    pivot_table = df.pivot_table(index=['index', 'arch', 'config', 'dataformat', 'model'], columns='build_id', values='samples_per_sec', aggfunc='max')
    df = pd.DataFrame(pivot_table.to_records())

    # add pct_diff column
    df.drop(columns=['index'], inplace=True)
    df['pct_diff'] = ((df[ids[1]] - df[ids[0]]) / df[ids[0]]) * 100

    return df

def print_diff(df: pd.DataFrame):
    """ Print the perf diff to console. """
    # format the pct_diff column
    def format_value(value):
        if not math.isnan(value):
            if value > 1:
                return f'\x1b[32m{value:.2f}%\x1b[0m'  # ANSI code for green color
            elif value < -1:
                return f'\x1b[31m{value:.2f}%\x1b[0m'  # ANSI code for red color
            else:
                return f'\x1b[37m{value:.2f}%\x1b[0m'  # ANSI code for red color
        else:
            return 'N/A'

    df['pct_diff'] = df['pct_diff'].apply(format_value)
    df = df.round(2)
    pd.set_option("display.max_rows", None)
    print(df)

def main():
    parser = argparse.ArgumentParser(description='Compare performance for two builds.')
    parser.add_argument('build_ids', nargs=2, help='Build IDs to compare')
    parser.add_argument('-o', '--output', help='Output file path (CSV format)', default=None)
    args = parser.parse_args()

    build_ids = args.build_ids

    # correct the build ids prefix
    # TODO: Refactor the code to work on whatever new performance comparision we use in the future.
    prefix = "gitlab-pipeline-"
    build_ids = [(x if x.startswith(prefix) else prefix + str(x)) for x in build_ids]

    # compare
    df = compare_perf(build_ids)

    # save to file
    if args.output:
        df.to_csv(args.output, index=False)

    print_diff(df)

if __name__ == "__main__":
    main()

# TODO: Refactor the code to work on whatever new performance comparision we use in the future.
def test_compare_perf():
    df = compare_perf(["gitlab-pipeline-479274", "gitlab-pipeline-479323"])
    print_diff(df)
