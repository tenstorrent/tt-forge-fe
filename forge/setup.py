# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Third Party
from setuptools import find_packages, setup

setup(
    name="forge",
    version="0.1",
    description="Tenstorrent Python Forge framework",
    packages=["forge"],
    package_dir={"forge": "forge"},
)
