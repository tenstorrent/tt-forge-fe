# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# import sys
from test.models.pytorch.vision.complex_yolov4.utils.darknet2pytorch import Darknet


def create_model(configs):
    """Create model based on architecture name"""
    if (configs.arch == "darknet") and (configs.cfgfile is not None):
        print("using darknet")
        model = Darknet(cfgfile=configs.cfgfile)
    else:
        assert False, "Undefined model backbone"

    return model
