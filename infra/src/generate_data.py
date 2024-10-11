# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
from loguru import logger
from utils import get_github_runner_environment
from cicd import create_cicd_json_for_data_analysis, get_cicd_json_filename


def create_pipeline_json(workflow_filename: str, jobs_filename: str, workflow_outputs_dir):

    github_runner_environment = get_github_runner_environment()
    pipeline = create_cicd_json_for_data_analysis(
        workflow_outputs_dir,
        github_runner_environment,
        workflow_filename,
        jobs_filename,
    )

    report_filename = get_cicd_json_filename(pipeline)
    logger.info(f"Writing pipeline JSON to {report_filename}")

    with open(report_filename, "w") as f:
        f.write(pipeline.json())

    return pipeline, report_filename


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True, help="Run ID of the workflow")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="generated/cicd",
        help="Output directory for the pipeline json",
    )
    args = parser.parse_args()

    logger.info(f"Creating pipeline JSON for workflow run ID {args.run_id}")
    create_pipeline_json(
        workflow_filename=f"{args.output_dir}/{args.run_id}/workflow.json",
        jobs_filename=f"{args.output_dir}/{args.run_id}/workflow_jobs.json",
        workflow_outputs_dir=args.output_dir,
    )
