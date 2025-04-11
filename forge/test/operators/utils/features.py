# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os


class TestFeaturesConfiguration:
    """Store test features configuration"""

    __test__ = False  # Disable pytest collection

    @staticmethod
    def get_env_property(env_var: str, default_value: str):
        return os.getenv(env_var, default_value)

    SKIP_FORGE_VERIFICATION = get_env_property("SKIP_FORGE_VERIFICATION", "false").lower() == "true"
    CAPTURE_OUTPUT = get_env_property("CAPTURE_OUTPUT", "false").lower() == "true"
    TRACE_XFAIL_VALIDATION = get_env_property("TRACE_XFAIL_VALIDATION", "false").lower() == "true"
    DRY_RUN = get_env_property("DRY_RUN", "false").lower() == "true"
    EMULATE_RUN = get_env_property("EMULATE_RUN", "false").lower() == "true"
    STATUS_TRACKER = get_env_property("STATUS_TRACKER", "false").lower() == "true"
    VERIFICATION_TIMEOUT = int(get_env_property("VERIFICATION_TIMEOUT", "300"))
