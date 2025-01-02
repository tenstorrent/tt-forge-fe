# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets.pipelines import Compose
from .loading import (LoadAnnotations3D, LoadImageFromFileMono3D,
                      LoadMultiViewImageFromFiles, LoadPointsFromFile,
                      LoadPointsFromMultiSweeps, NormalizePointsColor,
                      PointSegClassMapping)
