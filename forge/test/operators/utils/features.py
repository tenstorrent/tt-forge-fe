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
