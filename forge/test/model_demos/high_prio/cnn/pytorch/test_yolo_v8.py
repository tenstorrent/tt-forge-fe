import pytest
from PIL import Image
import os
import torch
import forge
from ultralytics import YOLO

# Function to load YOLO models
def load_yolo_models(weight_paths):
    models = {}
    for weights in weight_paths:
        try:
            model_name = weights.split('.')[0]
            models[model_name] = YOLO(weights)
            print(f"Loaded {model_name} model successfully!")
        except Exception as e:
            print(f"Error loading {weights}: {e}")
    return models

# Ultralytics version not supported

# # Function to load YOLO OBB models
# def load_yolo_obb_models(weight_paths):
#     models = {}
#     for weights in weight_paths:
#         try:
#             model_name = weights.split('.')[0]
#             models[model_name] = YOLO(weights)
#             print(f"Loaded {model_name} model successfully!")
#         except Exception as e:
#             print(f"Error loading {weights}: {e}")
#     return models

# Function to load YOLO pose models
def load_yolo_pose_models(weight_paths):
    models = {}
    for weights in weight_paths:
        try:
            model_name = weights.split('.')[0]
            models[model_name] = YOLO(weights)
            print(f"Loaded {model_name} model successfully!")
        except Exception as e:
            print(f"Error loading {weights}: {e}")
    return models

# Function to load YOLO Seg models
def load_yolo_seg_models(weight_paths):
    models = {}
    for weights in weight_paths:
        try:
            model_name = weights.split('.')[0]
            models[model_name] = YOLO(weights)
            print(f"Loaded {model_name} model successfully!")
        except Exception as e:
            print(f"Error loading {weights}: {e}")
    return models


# Function to load YOLO Cls models
def load_yolo_cls_models(weight_paths):
    models = {}
    for weights in weight_paths:
        try:
            model_name = weights.split('.')[0]
            models[model_name] = YOLO(weights)
            print(f"Loaded {model_name} model successfully!")
        except Exception as e:
            print(f"Error loading {weights}: {e}")
    return models

# List of model weights for regular YOLO models
weight_files = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
# Load the regular models
loaded_models = load_yolo_models(weight_files)
print("Loaded models:", loaded_models.keys())

# List of model weights for YOLO Seg models
weight_files_seg = ["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt", "yolov8x-seg.pt"]
# Load the Seg models
loaded_models_seg = load_yolo_seg_models(weight_files_seg)
print("Loaded Seg models:", loaded_models_seg.keys())

# List of model weights for YOLO pose models
weight_files_pose = ["yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt", "yolov8l-pose.pt", "yolov8x-pose.pt"]
# Load the pose models
loaded_models_pose = load_yolo_pose_models(weight_files_pose)
print("Loaded Pose models:", loaded_models_pose.keys())

# # List of model weights for YOLO OBB models (again with a different name set)
# weight_files_obb_cls = ["yolov8n-obb.pt", "yolov8s-obb.pt", "yolov8m-obb.pt", "yolov8l-obb.pt", "yolov8x-obb.pt"]
# # Load the OBB models (classification version)
# loaded_models_obb_cls = load_yolo_obb_models(weight_files_obb_cls)
# print("Loaded OBB classification models:", loaded_models_obb_cls.keys())

# List of model weights for YOLO classification models
weight_files_cls = ["yolov8n-cls.pt", "yolov8s-cls.pt", "yolov8m-cls.pt", "yolov8l-cls.pt", "yolov8x-cls.pt"]
# Load the classification models
loaded_models_cls = load_yolo_cls_models(weight_files_cls)
print("Loaded classification models:", loaded_models_cls.keys())
