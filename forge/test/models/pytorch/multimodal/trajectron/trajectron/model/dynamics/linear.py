# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from ..dynamics import Dynamic


class Linear(Dynamic):
    def init_constants(self):
        pass

    def integrate_samples(self, v, x):
        return v

    def integrate_distribution(self, v_dist, x):
        return v_dist
