#!/bin/bash
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
set -u
source $PYTHON_ENV/bin/activate
pip install sphinx
pip install sphinx-rtd-theme
pip install sphinx-markdown-builder
sphinx-build -M $BUILDER $SOURCE_DIR $INSTALL_DIR
