# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch

# from bevformer_test import BEVFormer
from bevformer_test import BEVFormer
from checkpoint import load_checkpoint
from nuscenes_dataloader import build_dataloader
from nuscenes_dataset import CustomNuScenesDataset
from registry import build_dataset

import forge

_ = CustomNuScenesDataset


def test_bevformer():

    compiler_cfg = forge.config._get_global_compiler_config()
    # compiler_cfg.enable_tvm_constant_prop = True
    # compiler_cfg.convert_framework_params_to_tvm = True
    # compiler_cfg.tvm_constnat_prop_mask = {"zeros.weight", "zeros.bias"}
    pts_bbox_head = {
        "type": "BEVFormerHead",
        "bev_h": 50,
        "bev_w": 50,
        "num_query": 900,
        "num_classes": 10,
        "in_channels": 256,
        "sync_cls_avg_factor": True,
        "with_box_refine": True,
        "as_two_stage": False,
        "transformer": {
            "type": "PerceptionTransformer",
            "rotate_prev_bev": True,
            "use_shift": True,
            "use_can_bus": True,
            "embed_dims": 256,
            "encoder": {
                "type": "BEVFormerEncoder",
                "num_layers": 3,
                "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                "num_points_in_pillar": 4,
                "return_intermediate": False,
                "transformerlayers": {
                    "type": "BEVFormerLayer",
                    "attn_cfgs": [
                        {"type": "TemporalSelfAttention", "embed_dims": 256, "num_levels": 1},
                        {
                            "type": "SpatialCrossAttention",
                            "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                            "deformable_attention": {
                                "type": "MSDeformableAttention3D",
                                "embed_dims": 256,
                                "num_points": 8,
                                "num_levels": 1,
                            },
                            "embed_dims": 256,
                        },
                    ],
                    "feedforward_channels": 512,
                    "ffn_dropout": 0.1,
                    "operation_order": ("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
                },
            },
            "decoder": {
                "type": "DetectionTransformerDecoder",
                "num_layers": 6,
                "return_intermediate": True,
                "transformerlayers": {
                    "type": "DetrTransformerDecoderLayer",
                    "attn_cfgs": [
                        {"type": "MultiheadAttention", "embed_dims": 256, "num_heads": 8, "dropout": 0.1},
                        {"type": "CustomMSDeformableAttention", "embed_dims": 256, "num_levels": 1},
                    ],
                    "feedforward_channels": 512,
                    "ffn_dropout": 0.1,
                    "operation_order": ("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
                },
            },
        },
        "bbox_coder": {
            "type": "NMSFreeCoder",
            "post_center_range": [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            "max_num": 300,
            "voxel_size": [0.2, 0.2, 8],
            "num_classes": 10,
        },
        "positional_encoding": {
            "type": "LearnedPositionalEncoding",
            "num_feats": 128,
            "row_num_embed": 50,
            "col_num_embed": 50,
        },
        "loss_cls": {"type": "FocalLoss", "use_sigmoid": True, "gamma": 2.0, "alpha": 0.25, "loss_weight": 2.0},
        "loss_bbox": {"type": "L1Loss", "loss_weight": 0.25},
        "loss_iou": {"type": "GIoULoss", "loss_weight": 0.0},
        "train_cfg": None,
        "test_cfg": None,
    }
    img_backbone = {
        "type": "ResNet",
        "depth": 50,
        "num_stages": 4,
        "out_indices": (3,),
        "frozen_stages": 1,
        "norm_cfg": {"type": "BN", "requires_grad": False},
        "norm_eval": True,
        "style": "pytorch",
    }
    img_neck = {
        "type": "FPN",
        "in_channels": [2048],
        "out_channels": 256,
        "start_level": 0,
        "add_extra_convs": "on_output",
        "num_outs": 1,
        "relu_before_extra_convs": True,
    }

    data_test = {
        "type": "CustomNuScenesDataset",
        "data_root": "data/nuscenes/",
        "ann_file": "data/nuscenes/nuscenes_infos_temporal_val.pkl",
        "pipeline": [
            {"type": "LoadMultiViewImageFromFiles", "to_float32": True},
            {
                "type": "NormalizeMultiviewImage",
                "mean": [123.675, 116.28, 103.53],
                "std": [58.395, 57.12, 57.375],
                "to_rgb": True,
            },
            {
                "type": "MultiScaleFlipAug3D",
                "img_scale": (1600, 900),
                "pts_scale_ratio": 1,
                "flip": False,
                "transforms": [
                    {"type": "RandomScaleImageMultiViewImage", "scales": [0.5]},
                    {"type": "PadMultiViewImage", "size_divisor": 32},
                    {
                        "type": "DefaultFormatBundle3D",
                        "class_names": [
                            "car",
                            "truck",
                            "construction_vehicle",
                            "bus",
                            "trailer",
                            "barrier",
                            "motorcycle",
                            "bicycle",
                            "pedestrian",
                            "traffic_cone",
                        ],
                        "with_label": False,
                    },
                    {"type": "CustomCollect3D", "keys": ["img"]},
                ],
            },
        ],
        "classes": [
            "car",
            "truck",
            "construction_vehicle",
            "bus",
            "trailer",
            "barrier",
            "motorcycle",
            "bicycle",
            "pedestrian",
            "traffic_cone",
        ],
        "modality": {
            "use_lidar": False,
            "use_camera": True,
            "use_radar": False,
            "use_map": False,
            "use_external": True,
        },
        "test_mode": True,
        "box_type_3d": "LiDAR",
        "bev_size": (50, 50),
    }
    dataset = build_dataset(data_test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=4,
        dist=True,
        shuffle=False,
        nonshuffler_sampler={"type": "DistributedSampler"},
    )
    for k in data_loader:
        input_image = k
        break
    model = BEVFormer(
        img_backbone=img_backbone,
        pts_bbox_head=pts_bbox_head,
        img_neck=img_neck,
        use_grid_mask=True,
        video_test_mode=True,
    )
    checkpoint = load_checkpoint(model, "./bevformer_tiny_epoch_24.pth", map_location="cpu")
    model.eval()
    # print(model)

    img_metas_data_container = input_image["img_metas"][0]
    img_metas = img_metas_data_container.data
    img_data_container = input_image["img"][0]
    filename = img_metas[0][0]["filename"]
    ori_shapes = img_metas[0][0]["ori_shape"]
    img_shapes = img_metas[0][0]["img_shape"]
    lidar2img = img_metas[0][0]["lidar2img"]
    lidar2cam = img_metas[0][0]["lidar2cam"]
    pad_shape = img_metas[0][0]["pad_shape"]
    box_mode_3d = img_metas[0][0]["box_mode_3d"]
    box_type_3d = img_metas[0][0]["box_type_3d"]
    img_norm_cfg = img_metas[0][0]["img_norm_cfg"]
    pts_filename = img_metas[0][0]["pts_filename"]
    can_bus = img_metas[0][0]["can_bus"]
    img = img_data_container.data
    ori_shapes_tensor = torch.tensor(ori_shapes, dtype=torch.float32).unsqueeze(0)
    img_shapes_tensor = torch.tensor(img_shapes, dtype=torch.float32).unsqueeze(0)
    pad_shapes_tensor = torch.tensor(pad_shape, dtype=torch.float32).unsqueeze(0)
    lidar2img_tensor = [torch.tensor(arr, dtype=torch.float32) for arr in lidar2img]
    lidar2img_stacked_tensor = torch.stack(lidar2img_tensor, dim=0).unsqueeze(0)
    lidar2cam_tensor = [torch.tensor(arr, dtype=torch.float32) for arr in lidar2cam]
    lidar2cam_stacked_tensor = torch.stack(lidar2cam_tensor, dim=0).unsqueeze(0)
    img_pybuda = img[0].unsqueeze(0)
    input_image_dict = {"rescale": True, "img_metas": img_metas, "img": img}

    # for i, data in enumerate(data_loader):
    #     with torch.no_grad():
    #         inputs = {}
    #         inputs['img'] = data['img'][0].data[0].float().unsqueeze(0)
    #         inputs['img_metas'] = [1]
    #         inputs['img_metas'][0] = [1]
    #         inputs['img_metas'][0][0] = {}
    #         inputs['img_metas'][0][0]['can_bus'] = torch.from_numpy(data['img_metas'][0].data[0][0]['can_bus'])
    #         inputs['img_metas'][0][0]['lidar2img'] = torch.from_numpy(np.array(data['img_metas'][0].data[0][0]['lidar2img']))
    #         inputs['img_metas'][0][0]['scene_token'] = 'fcbccedd61424f1b85dcbf8f897f9754'
    #         inputs['img_metas'][0][0]['img_shape'] = torch.Tensor([[480, 800]])
    #         img = inputs['img']
    #         img_metas = inputs['img_metas']
    #         output_file = 'bevformer_without_nms.onnx'
    #         torch.onnx.register_custom_op_symbolic("aten::lift_fresh", lambda g, x: x, 11)
    #         torch.onnx.export(
    #             model,
    #             (img, img_metas),
    #             output_file,
    #             verbose=True,
    #             opset_version=11,
    #         )
    #         break

    # print("input_image_dict ",input_image_dict)
    # result = model(return_loss=False,**input_image_dict)
    # print(result)
    # exit()
    class BEV_wrapper(torch.nn.Module):
        def __init__(self, model, filename, box_mode_3d, box_type_3d, img_norm_cfg, pts_filename, can_bus):
            super().__init__()
            self.model = model
            self.filename = filename
            self.box_mode_3d = box_mode_3d
            self.box_type_3d = box_type_3d
            self.img_norm_cfg = img_norm_cfg
            self.pts_filename = pts_filename
            self.can_bus = can_bus

        def forward(
            self,
            ori_shapes_tensor,
            img_shapes_tensor,
            lidar2img_stacked_tensor,
            lidar2cam_stacked_tensor,
            pad_shapes_tensor,
            img_pybuda,
        ):
            lidar2img_stacked_tensor = lidar2img_stacked_tensor.squeeze(0)
            lidar2cam_stacked_tensor = lidar2cam_stacked_tensor.squeeze(0)
            ori_shapes_tensor = ori_shapes_tensor.squeeze(0)
            img_shapes_tensor = img_shapes_tensor.squeeze(0)
            pad_shapes_tensor = pad_shapes_tensor.squeeze(0)
            img_pybuda = img_pybuda.squeeze(0)
            lidar2img_array = [tensor.numpy() for tensor in lidar2img_stacked_tensor]
            lidar2cam_array = [tensor.numpy() for tensor in lidar2cam_stacked_tensor]
            ori_shapes_list = [tuple(row.tolist()) for row in ori_shapes_tensor]
            img_shapes_list = [tuple(row.tolist()) for row in img_shapes_tensor]
            pad_shapes_list = [tuple(row.tolist()) for row in pad_shapes_tensor]
            img_metas = {
                "filename": self.filename,
                "ori_shape": ori_shapes_list,
                "img_shape": img_shapes_list,
                "lidar2img": lidar2img_array,
                "lidar2cam": lidar2cam_array,
                "pad_shape": pad_shapes_list,
                "scale_factor": 1.0,
                "flip": False,
                "pcd_horizontal_flip": False,
                "pcd_vertical_flip": False,
                "box_mode_3d": self.box_mode_3d,
                "box_type_3d": self.box_type_3d,
                "img_norm_cfg": self.img_norm_cfg,
                "sample_idx": "3e8750f331d7499e9b5123e9eb70f2e2",
                "prev_idx": "",
                "next_idx": "3950bd41f74548429c0f7700ff3d8269",
                "pcd_scale_factor": 1.0,
                "pts_filename": self.pts_filename,
                "scene_token": "fcbccedd61424f1b85dcbf8f897f9754",
                "can_bus": self.can_bus,
            }
            input_pybuda_dict = {"rescale": True, "img_metas": [[img_metas]], "img": [img_pybuda]}
            # print("input_pybuda_dict = ",input_pybuda_dict)
            output = self.model(return_loss=False, **input_pybuda_dict)
            # breakpoint()
            # boxes_3d_tensor = output[0]['pts_bbox']['boxes_3d'].tensor
            # output[0]['pts_bbox']['boxes_3d'] = boxes_3d_tensor

            # boxes_3d_tensor = output[0]['pts_bbox']['boxes_3d']
            # scores_3d_tensor = output[0]['pts_bbox']['scores_3d']
            # labels_3d_tensor = output[0]['pts_bbox']['labels_3d']
            # output = (boxes_3d_tensor.detach(), scores_3d_tensor.detach(), labels_3d_tensor)
            # logger.info(f"outputs = {output['all_bbox_preds']}")
            # return output['all_bbox_preds']
            return output

    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.convert_framework_params_to_tvm = True
    wrapper_bev_model = BEV_wrapper(model, filename, box_mode_3d, box_type_3d, img_norm_cfg, pts_filename, can_bus)
    wrapper_bev_model.eval()
    # result_wrapper = wrapper_bev_model(ori_shapes_tensor,img_shapes_tensor,lidar2img_stacked_tensor,lidar2cam_stacked_tensor,pad_shapes_tensor, img_pybuda)
    # traced_model = torch.jit.trace(wrapper_bev_model, (ori_shapes_tensor, img_shapes_tensor, lidar2img_stacked_tensor, lidar2cam_stacked_tensor, pad_shapes_tensor, img_pybuda),strict= False)
    # result_jit_trace = traced_model(ori_shapes_tensor, img_shapes_tensor, lidar2img_stacked_tensor, lidar2cam_stacked_tensor, pad_shapes_tensor, img_pybuda)
    # result_wrapper = wrapper_bev_model(ori_shapes_tensor,img_shapes_tensor,lidar2img_stacked_tensor,lidar2cam_stacked_tensor,pad_shapes_tensor, img_pybuda)
    # print("results jit trace = ",result_jit_trace)
    # print("result_wrapper = ",result_wrapper)
    compiled_model = forge.compile(
        wrapper_bev_model,
        sample_inputs=[
            ori_shapes_tensor,
            img_shapes_tensor,
            lidar2img_stacked_tensor,
            lidar2cam_stacked_tensor,
            pad_shapes_tensor,
            img_pybuda,
        ],
        module_name="tsa",
    )


if __name__ == "__main__":
    test_bevformer()
