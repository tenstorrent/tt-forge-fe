import torch
from mmcv import Config
import sys
sys.path.append('forge/test/models/pytorch/vision/PETR')
from mmdet3d.models.builder import build_model
from utils.petr3d import Petr3D
from utils.petr_head import PETRHead
from utils.hungarian_assigner_3d import HungarianAssigner3D
from utils.match_cost import BBox3DL1Cost
from mmdet.core.bbox.samplers import pseudo_sampler
from mmdet.models.losses import focal_loss,iou_loss
from mmdet.models.losses.smooth_l1_loss import L1Loss
from mmdet.core.bbox.coder import distance_point_bbox_coder
from utils.positional_encoding import SinePositionalEncoding3D
from utils.petr_transformer import PETRTransformer
from utils.nms_free_coder import NMSFreeClsCoder
from mmdet.models.backbones.resnet import ResNet
from utils.grid_mask import GridMask
from mmcv.runner import load_checkpoint
from utils.nuscenes_dataset import CustomNuScenesDataset
from mmdet3d.datasets import build_dataloader, build_dataset
from utils.transform_3d import ResizeCropFlipImage
from mmdet3d.datasets.pipelines.test_time_aug import MultiScaleFlipAug3D
from mmdet3d.datasets.pipelines.formating import DefaultFormatBundle3D
from mmcv.parallel import MMDataParallel
from utils.cp_fpn import CPFPN
from utils.vovnetcp import VoVNetCP

import pytest 
import forge


from loguru import logger 

# variants = ["petr_r50dcn_gridmask_c5","petr_r50dcn_gridmask_p4","petr_vovnet_gridmask_p4_800x320","petr_vovnet_gridmask_p4_1600x640"]
# variants = ["petr_r50dcn_gridmask_c5"]
variants = ["petr_vovnet_gridmask_p4_800x320"]# "petr_vovnet_gridmask_p4_1600x640"]
@pytest.mark.parametrize("variant", variants)
def test_petr(variant):

    # model load
    cfg = Config.fromfile(f'/proj_sw/user_dev/kkannan/dec30_forge/tt-forge-fe/forge/test/models/pytorch/vision/PETR/utils/{variant}.py')
    cfg.data.test.test_mode = True
    cfg.model.pretrained = None
    cfg.data.test.ann_file = '/proj_sw/user_dev/kkannan/dec30_forge/tt-forge-fe/forge/test/models/pytorch/vision/PETR/data/nuscenes/nuscenes_infos_val.pkl'
    model = build_model(cfg.model,test_cfg=cfg.get('test_cfg'),train_cfg=cfg.get('train_cfg'))
    checkpoint = load_checkpoint(model, f'forge/test/models/pytorch/vision/PETR/weights/{variant}.pth', map_location='cpu')

    # prepare input
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False)
    

    model = MMDataParallel(model, device_ids=[0])
    
    logger.info("Model is running on device={}",next(model.parameters()).device)
    model.eval()
    
    dataset = data_loader.dataset


    
    for i, data in enumerate(data_loader):
        
        # logger.info("input nature->{}",data)
        
        img_metas = data['img_metas'][0].data[0]
        
        filename = img_metas[0]['filename']
        ori_shape = img_metas[0]['ori_shape']
        img_shape = img_metas[0]['img_shape']
        pad_shape = img_metas[0]['pad_shape']
        scale_factor = img_metas[0]['scale_factor']
        flip = img_metas[0]['flip']
        pcd_horizontal_flip = img_metas[0]['pcd_horizontal_flip']
        pcd_vertical_flip = img_metas[0]['pcd_vertical_flip']
        box_mode_3d = img_metas[0]['box_mode_3d']
        box_type_3d = img_metas[0]['box_type_3d']
        mean = torch.from_numpy(img_metas[0]['img_norm_cfg']['mean'])
        std = torch.from_numpy(img_metas[0]['img_norm_cfg']['std'])
        to_rgb = img_metas[0]['img_norm_cfg']['to_rgb']
        sample_idx = img_metas[0]['sample_idx']
        pcd_scale_factor = img_metas[0]['pcd_scale_factor']
        pts_filename = img_metas[0]['pts_filename']
        img = data['img'][0].data[0]
        lidar2img_list = img_metas[0]['lidar2img'] 
        
        lidar2img_tensors_list = []
  
        for idx, lidar2img_array in enumerate(lidar2img_list):
            lidar2img_tensor = torch.from_numpy(lidar2img_array)
            lidar2img_tensors_list.append(lidar2img_tensor)
           
    
    class petr_wrapper(torch.nn.Module):
        
        def __init__(self,model,filename,ori_shape,img_shape,pad_shape,scale_factor,flip,pcd_horizontal_flip,pcd_vertical_flip,box_mode_3d,box_type_3d,to_rgb, sample_idx,pcd_scale_factor,pts_filename ):
            super().__init__()
            self.model = model
                
            self.filename = filename
            self.ori_shape = ori_shape
            self.img_shape = img_shape
            self.pad_shape = pad_shape
            self.scale_factor = scale_factor
            self.flip = flip
            self.pcd_horizontal_flip = pcd_horizontal_flip
            self.pcd_vertical_flip = pcd_vertical_flip
            self.box_mode_3d = box_mode_3d
            self.box_type_3d = box_type_3d
            self.to_rgb = to_rgb
            self.sample_idx = sample_idx
            self.pcd_scale_factor = pcd_scale_factor
            self.pts_filename = pts_filename
            
        def forward (self,l0,l1,l2,l3,l4,l5,img,mean,std):
                    
            l0 = l0.squeeze(0)
            l1 = l1.squeeze(0)
            l2 = l2.squeeze(0)
            l3 = l3.squeeze(0)
            l4 = l4.squeeze(0)
            l5 = l5.squeeze(0)
            img = img.squeeze(0)
            mean = mean.squeeze(0)
            std = std.squeeze(0)
            
            data = {
                'img_metas': [[{
                    'filename': self.filename,
                    'ori_shape': self.ori_shape,
                    'img_shape': self.img_shape,
                    'lidar2img': [l0,l1,l2,l3,l4,l5],
                    'pad_shape': self.pad_shape,
                    'scale_factor': self.scale_factor,
                    'flip': self.flip,
                    'pcd_horizontal_flip': self.pcd_horizontal_flip,
                    'pcd_vertical_flip': self.pcd_vertical_flip,
                    'box_mode_3d': self.box_mode_3d,
                    'box_type_3d': self.box_type_3d,
                    'img_norm_cfg': {
                        'mean': mean,
                        'std': std,
                        'to_rgb': self.to_rgb
                    },
                    'sample_idx': self.sample_idx,
                    'pcd_scale_factor': self.pcd_scale_factor,
                    'pts_filename': self.pts_filename
                }]],
                'img': [img]
            }

            output = model(**data)
            
            boxes_3d_tensor = output[0]['pts_bbox']['boxes_3d'].tensor
            scores_3d_tensor = output[0]['pts_bbox']['scores_3d']
            labels_3d_tensor = output[0]['pts_bbox']['labels_3d']
            output = (boxes_3d_tensor, scores_3d_tensor, labels_3d_tensor)
            
            return output
            
            
    wrapped_model = petr_wrapper(model,filename,ori_shape,img_shape,pad_shape,scale_factor,flip,pcd_horizontal_flip,pcd_vertical_flip,box_mode_3d,box_type_3d,to_rgb, sample_idx,pcd_scale_factor,pts_filename)
    
    wrapped_model.eval()
    
    
    
    
    inputs = [lidar2img_tensors_list[0].unsqueeze(0),lidar2img_tensors_list[1].unsqueeze(0),lidar2img_tensors_list[2].unsqueeze(0),lidar2img_tensors_list[3].unsqueeze(0),lidar2img_tensors_list[4].unsqueeze(0),lidar2img_tensors_list[5].unsqueeze(0),img.unsqueeze(0),mean.unsqueeze(0),std.unsqueeze(0)]
    
    
    for param in model.parameters():
        param.requires_grad = False
        
    # logger.info("model={}\n",model)
        
    
    
    # with torch.no_grad():
    #     result = wrapped_model(lidar2img_tensors_list[0].unsqueeze(0),lidar2img_tensors_list[1].unsqueeze(0),lidar2img_tensors_list[2].unsqueeze(0),lidar2img_tensors_list[3].unsqueeze(0),lidar2img_tensors_list[4].unsqueeze(0),lidar2img_tensors_list[5].unsqueeze(0),img.unsqueeze(0),mean.unsqueeze(0),std.unsqueeze(0))
        
    # logger.info("\n=============CPU inference OUTPUTS ==================")
    # logger.info("outputs={}", result)
    # logger.info("==============================================\n")
    # logger.info("\n=============FORGE inference .... ==================")
        
        
    # logger.info("local trace")
    # with torch.no_grad():
    #     op = torch.jit.trace(wrapped_model, (lidar2img_tensors_list[0].unsqueeze(0),lidar2img_tensors_list[1].unsqueeze(0),lidar2img_tensors_list[2].unsqueeze(0),lidar2img_tensors_list[3].unsqueeze(0),lidar2img_tensors_list[4].unsqueeze(0),lidar2img_tensors_list[5].unsqueeze(0),img.unsqueeze(0),mean.unsqueeze(0),std.unsqueeze(0)))
        
    # # logger.info("\n=============local jit trace done!!! ==================")
        
    # with torch.no_grad():
    
    compiled_model = forge.compile(wrapped_model, sample_inputs=inputs, module_name=f"pt_{variant}")
    
  