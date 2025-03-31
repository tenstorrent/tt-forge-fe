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
from utils.util import denormalize_bbox
from mmdet3d.core.bbox.transforms import bbox3d2result

import pytest 
import forge
from forge.verify.verify import verify
from forge.verify.config import DepricatedVerifyConfig


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
            
            all_cls_scores = output['all_cls_scores']
            all_bbox_preds= output['all_bbox_preds']
            
            output = (all_cls_scores, all_bbox_preds)
            
            
            return output
     
       
    wrapped_model = petr_wrapper(model,filename,ori_shape,img_shape,pad_shape,scale_factor,flip,pcd_horizontal_flip,pcd_vertical_flip,box_mode_3d,box_type_3d,to_rgb, sample_idx,pcd_scale_factor,pts_filename)
    
    wrapped_model.eval()
    
    # logger.info("sliced model 2={}",wrapped_model)

    # for i, tensor in enumerate(lidar2img_tensors_list):
    #     print(f"Shape of lidar2img_tensors_list[{i}] before unsqueeze: {tensor.shape}")

        
    # print("img.shape={}",img.shape)
    # print("mean.shape={}",mean.shape)
    # print("std.shape{}",std.shape)
    
   
    inputs = [lidar2img_tensors_list[0].unsqueeze(0),lidar2img_tensors_list[1].unsqueeze(0),lidar2img_tensors_list[2].unsqueeze(0),lidar2img_tensors_list[3].unsqueeze(0),lidar2img_tensors_list[4].unsqueeze(0),lidar2img_tensors_list[5].unsqueeze(0),img.unsqueeze(0),mean.unsqueeze(0),std.unsqueeze(0)]
    
        
    for i, tensor in enumerate(inputs):
        print(f"Data type of inputs[{i}] : {tensor.dtype}")
        if tensor.dtype == torch.float64:
            inputs[i] = tensor.to(torch.float32)
            
    for i, tensor in enumerate(inputs):
        print(f"Data type of inputs [{i}] after conversion: {tensor.dtype}")
        
    # logger.info("inputs={}",inputs)
    
    for param in model.parameters():
        param.requires_grad = False
        
        
    # logger.info("model={}\n",model)
    
   
    # with torch.no_grad():
    #     result = wrapped_model(lidar2img_tensors_list[0].unsqueeze(0),lidar2img_tensors_list[1].unsqueeze(0),lidar2img_tensors_list[2].unsqueeze(0),lidar2img_tensors_list[3].unsqueeze(0),lidar2img_tensors_list[4].unsqueeze(0),lidar2img_tensors_list[5].unsqueeze(0),img.unsqueeze(0),mean.unsqueeze(0),std.unsqueeze(0))
        
    
    # logger.info("\n=============FORGE inference with  ================== (verify_forge_codegen_vs_framework=False")
    
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.convert_framework_params_to_tvm = True
    
    
    ompiled_model = forge.compile(wrapped_model, sample_inputs=inputs, module_name=f"pt_{variant}_jan16")
    
    
    
    # compiled_model = forge.compile(wrapped_model, sample_inputs=inputs, module_name=f"pt_{variant}", verify_cfg=DepricatedVerifyConfig(verify_forge_codegen_vs_framework=False))
    # verify_cfg.verify_forge_codegen_vs_framework
    
    all_cls_scores_1, all_bbox_pred_1 = result
    
    
    def decode_single(cls_scores, bbox_preds):
        
        # post processing 
        
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        voxel_size =[0.2, 0.2, 8]
        post_center_range =[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
        max_num = 300
        score_threshold =None
        num_classes =10
                

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % num_classes
        bbox_index = indexs // num_classes
        
        logger.info("cls_scores.shape={}",cls_scores.shape)
        logger.info("bbox_preds.shape={}",bbox_preds.shape)
        
        logger.info("cls_scores={}",cls_scores)
        logger.info("bbox_preds={}",bbox_preds)
        
        logger.info("indexs={}",indexs)
        logger.info("indexs.shape={}",indexs.shape)
            
    
        
        bbox_preds = bbox_preds[bbox_index]

        final_box_preds = denormalize_bbox(bbox_preds, pc_range)   
        final_scores = scores 
        final_preds = labels 

        # use score threshold
        if score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
        if post_center_range is not None:
            post_center_range = torch.tensor(post_center_range, device=scores.device)
            
            mask = (final_box_preds[..., :3] >=
                    post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <=
                        post_center_range[3:]).all(1)

            if score_threshold:
                mask &= thresh_mask
                

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels
            }
            
            return predictions_dict 
            
            
    # decode 
            
    all_cls_scores = all_cls_scores_1[-1]
    all_bbox_preds = all_bbox_pred_1[-1]
    
    batch_size = all_cls_scores.size()[0]
    predictions_list = []
    for i in range(batch_size):
        predictions_list.append(decode_single(all_cls_scores[i], all_bbox_preds[i]))
        
    
    # petr head steps 
    
    preds_dicts = predictions_list
    
    num_samples = len(preds_dicts)

    ret_list = []
    for i in range(num_samples):
        preds = preds_dicts[i]
        bboxes = preds['bboxes']
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
        bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
        scores = preds['scores']
        labels = preds['labels']
        ret_list.append([bboxes, scores, labels])
       
       
    # petr3d simple_test_pts steps  
    
    bbox_list = ret_list
    
    bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ] 
        
    bbox_list = [dict() for i in range(1)]
    bbox_pts = bbox_results
    for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
        result_dict['pts_bbox'] = pts_bbox
        
    logger.info("bbox_list={}",bbox_list)
        
    output = bbox_list
    boxes_3d_tensor = output[0]['pts_bbox']['boxes_3d'].tensor
    scores_3d_tensor = output[0]['pts_bbox']['scores_3d']
    labels_3d_tensor = output[0]['pts_bbox']['labels_3d']

    my_output = (boxes_3d_tensor, scores_3d_tensor, labels_3d_tensor)
    
    logger.info("\n=============CPU inference OUTPUTS with external PP ==================")
    logger.info("my_output={}",my_output)
    logger.info("==============================================\n")
   
    
    







