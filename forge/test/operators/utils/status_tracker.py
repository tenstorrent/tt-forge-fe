# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Helper functions for tracking the status of sweep tests.


from datetime import datetime
import json
import os

from .plan import TestVector


class StatusTracker:

    counter = 0
    pid = os.getpid()

    def __init__(self, test_vector: TestVector, enabled: bool = False):
        self.test_vector = test_vector
        self.enabled = enabled
        self.status = {
            "pid": self.pid,
            "counter": StatusTracker.counter,
            "operator": self.test_vector.operator,
            "test_id": self.test_vector.get_id(),
            "status": None,
            "start_time": None,
            "end_time": None,
        }

    def store_status(self):
        file_name = f"test_status.json"
        with open(file_name, "w") as file:
            json.dump(self.status, file, indent=4)

    def __enter__(self):
        if not self.enabled:
            return
        self.status["status"] = "in progress"
        self.status["start_time"] = datetime.now().isoformat()
        self.store_status()
        StatusTracker.counter += 1

    def __exit__(self, exctype, excinst, exctb):
        if not self.enabled:
            return
        self.status["status"] = "completed"
        self.status["end_time"] = datetime.now().isoformat()
        self.store_status()
