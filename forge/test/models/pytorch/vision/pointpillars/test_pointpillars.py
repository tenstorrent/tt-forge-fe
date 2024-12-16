# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import os
from torchvision.ops import FeaturePyramidNetwork
from collections import OrderedDict
from custom.runner import Runner
from custom.imports import Config, ConfigDict, DictAction
from custom.HardVFE import PointPillarsScatter
from loguru import logger 
import forge


def test_pointpillars_pytorch(test_device):
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH
    
    # Load model
    config_path = "forge/test/models/pytorch/vision/pointpillars/configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb2-amp-2x_nus-3d.py"
    cfg = Config.fromfile(config_path)
    cfg.work_dir = 'forge/test/models/pytorch/vision/pointpillars/work_dirs/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d'
    checkpoint_path = 'forge/test/models/pytorch/vision/pointpillars/hv_pointpillars_secfpn_sbn-all_fp16_2x8_2x_nus-3d_20201020_222626-c3f0483e.pth'
    cfg.load_from = checkpoint_path
    runner = Runner.from_cfg(cfg) 
    model = runner.model
    
    points = torch.randn(1, 33587, 4, device='cpu')
    num_points = torch.randint(1, 100, (1, 4352,), device='cpu', dtype=torch.int32).to(torch.float32)
    voxels = torch.randn(1, 4352, 64, 4, device='cpu')
    voxel_centers = torch.randn(1, 4352, 3, device='cpu')
    coors = torch.randint(0, 200, (1, 4352, 4), device='cpu', dtype=torch.int32).to(torch.float32)           
    data_loader = runner.test_dataloader
    for i in data_loader:
        data_samples = i['data_samples']
        break
    
    class Point_wrapper(torch.nn.Module):
        def __init__(self, model, points, voxel_centers, data_samples):
            super().__init__()
            self.model = model
            self.data_samples = data_samples
            self.points = points
            self.voxel_centers = voxel_centers
        def forward(self, voxels, num_points, coors):
            points = self.points.squeeze(0)
            voxels = voxels.squeeze(0)
            voxel_centers = self.voxel_centers.squeeze(0)
            num_points = num_points.squeeze(0)
            coors = coors.squeeze(0)
            random_inputs = {
                'points': [points],
                'voxels': {
                    'num_points': num_points,  
                    'voxel_centers': voxel_centers, 
                    'voxels': voxels,  
                    'coors': coors
                }
            }
            output = self.model(random_inputs, self.data_samples, mode='predict')
            return output
    
    wrapper_bev_model = Point_wrapper(model, points, voxel_centers, data_samples)   
    wrapper_bev_model.eval()
    compiled_model = forge.compile(wrapper_bev_model, sample_inputs=[voxels, num_points, coors], module_name="pt_pointpillars")
