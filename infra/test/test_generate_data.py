# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from generate_data import create_pipeline_json
import os


def test_create_pipeline_json():
    """
    End-to-end test for create_pipeline_json function
    Calling this will generate a pipeline json file
    """
    os.environ["GITHUB_EVENT_NAME"] = "test"
    pipeline, filename = create_pipeline_json(
        workflow_filename="test/data/11236784732/workflow.json",
        jobs_filename="test/data/11236784732/workflow_jobs.json",
        workflow_outputs_dir="test/data",
    )

    assert pipeline is not None
    assert os.path.exists(filename)
