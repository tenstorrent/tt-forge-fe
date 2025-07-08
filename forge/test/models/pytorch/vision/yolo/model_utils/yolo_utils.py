# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from datasets import load_dataset
from torchvision import transforms
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.head import Detect
from pathlib import Path
import requests
import yaml
import yolov5
import cv2
import numpy as np
from PIL import Image
from yolov5.models.common import Detections
from yolov5.utils.dataloaders import exif_transpose, letterbox
from yolov5.utils.general import Profile, non_max_suppression, scale_boxes
from test.utils import fetch_model, yolov5_loader
from third_party.tt_forge_models.tools.utils import get_file

base_url = "https://github.com/ultralytics/yolov5/releases/download/v7.0"

class YoloWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.model[-1].end2end = False  # Disable internal post processing steps

    def forward(self, image: torch.Tensor):
        y, x = self.model(image)
        # Post processing inside model casts output to float32, even though raw output is aligned with image.dtype
        # Therefore we need to cast it back to image.dtype
        return (y.to(image.dtype), *x)


def load_yolo_model_and_image(url):
    # Load YOLO model weights
    weights = torch.hub.load_state_dict_from_url(url, map_location="cpu")

    # Initialize and load model
    model = DetectionModel(cfg=weights["model"].yaml)
    model.load_state_dict(weights["model"].float().state_dict())
    model.eval()

    # Load sample image and preprocess
    dataset = load_dataset("huggingface/cats-image", split="test[:1]")
    image = dataset[0]["image"]
    preprocess = transforms.Compose(
        [
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ]
    )
    image_tensor = preprocess(image).unsqueeze(0)

    return model, image_tensor


def postprocess(y):
    
    processed = Detect.postprocess(y.permute(0, 2, 1), 50)
    
    # URL to COCO class names YAML (used by Ultralytics YOLO models)
    yaml_url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/coco.yaml"

    response = requests.get(yaml_url)
    coco_yaml = yaml.safe_load(response.text)
    class_names = coco_yaml['names']
    det = processed[0] 
    

    print("Detections:")
    for d in det:
        x, y, w, h, score, cls = d.tolist()
        cls = int(cls)
        label = class_names[cls] if cls < len(class_names) else f"Unknown({cls})"
        print(f"  Box: [x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}], Score: {score:.2f}, Class: {label} ({cls})")



def load_yolov5_model_and_image(variant, size , input_size):
    
    name = "yolov5" + size
    model = fetch_model(name, f"{base_url}/{name}.pt", yolov5_loader, variant=variant)
    image_path = get_file("http://images.cocodataset.org/val2017/000000397133.jpg")
    image_sample = cv2.imread(str(image_path))
    image_sample = Image.fromarray(np.uint8(image_sample)).convert("RGB")
    ims, n, files, shape0, shape1, pixel_values = data_preprocessing(image_sample, size=(input_size, input_size))
    return model, ims, n, files, shape0, shape1, pixel_values

def data_preprocessing(ims: Image.Image, size: tuple) -> tuple:
    """Data preprocessing function for YOLOv5 object detection.

    Parameters
    ----------
    ims : Image.Image
        Input image
    size : tuple
        Desired image size

    Returns
    -------
    tuple
        List of images, number of samples, filenames, image size, inference size, preprocessed images
    """

    if not isinstance(ims, (list, tuple)):
        ims = [ims]
    num_images = len(ims)
    shape_orig, shape_infer, filenames = [], [], []

    for idx, img in enumerate(ims):
        filename = getattr(img, "filename", f"image{idx}")
        img = np.asarray(exif_transpose(img))
        filename = Path(filename).with_suffix(".jpg").name
        filenames.append(filename)

        if img.shape[0] < 5:
            img = img.transpose((1, 2, 0))

        if img.ndim == 3:
            img = img[..., :3]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        shape_orig.append(img.shape[:2])
        scale = max(size) / max(img.shape[:2])
        shape_infer.append([int(dim * scale) for dim in img.shape[:2]])
        ims[idx] = img if img.flags["C_CONTIGUOUS"] else np.ascontiguousarray(img)

    shape_infer = [size[0] for _ in np.array(shape_infer).max(0)]
    imgs_padded = [letterbox(img, shape_infer, auto=False)[0] for img in ims]
    imgs_padded = np.ascontiguousarray(np.array(imgs_padded).transpose((0, 3, 1, 2)))
    tensor_imgs = torch.from_numpy(imgs_padded) / 255

    return ims, num_images, filenames, shape_orig, shape_infer, tensor_imgs


def data_postprocessing(
    ims: list,
    x_shape: torch.Size,
    pred: list,
    model: yolov5.models.common.AutoShape,
    n: int,
    shape0: list,
    shape1: list,
    files: list,
) -> Detections:
    """Data postprocessing function for YOLOv5 object detection.

    Parameters
    ----------
    ims : list
        List of input images
    x_shape : torch.Size
        Shape of each image
    pred : list
        List of model predictions
    model : yolov5.models.common.AutoShape
        Model
    n : int
        Number of input samples
    shape0 : list
        Image shape
    shape1 : list
        Inference shape
    files : list
        Filenames

    Returns
    -------
    Detections
        Detection object
    """

    # Create dummy dt tuple (not used but required for Detections)
    dt = (Profile(), Profile(), Profile())

    # Perform NMS
    y = non_max_suppression(
        prediction=pred,
        conf_thres=model.conf,
        iou_thres=model.iou,
        classes=None,
        agnostic=model.agnostic,
        multi_label=model.multi_label,
        labels=(),
        max_det=model.max_det,
    )

    # Scale bounding boxes
    for i in range(n):
        scale_boxes(shape1, y[i][:, :4], shape0[i])

    # Return Detections object
    return Detections(ims, y, files, times=dt, names=model.names, shape=x_shape)