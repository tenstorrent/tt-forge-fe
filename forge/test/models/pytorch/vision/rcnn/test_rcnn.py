# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import cv2
import pytest
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify


# Paper - https://arxiv.org/abs/1311.2524
# Repo - https://github.com/object-detection-algorithm/R-CNN
@pytest.mark.nightly
def test_rcnn_pytorch(forge_property_recorder):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH, model="rcnn", source=Source.TORCHVISION, task=Task.OBJECT_DETECTION
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Load Alexnet Model
    framework_model = torchvision.models.alexnet(pretrained=True)
    num_classes = 2
    num_features = framework_model.classifier[6].in_features

    # Create class specific linear SVMs [Refer Section 2 in paper]
    svm_layer = nn.Linear(num_features, num_classes)

    # Replacing the Alexnet's ImageNet specific 1000-way classification layer with a randomly initialized (N + 1)-way classification layer(where N is the number of object classes, plus 1 for background)
    # [Refer Section 2.3.Domain-specific fine-tuning in Paper]
    init.normal_(svm_layer.weight, mean=0, std=0.01)
    init.constant_(svm_layer.bias, 0)
    framework_model.classifier[6] = svm_layer

    framework_model.eval()

    # Image
    img = cv2.imread("forge/test/models/files/samples/images/car.jpg")

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((227, 227)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Selective search - A separate tool for generating proposals(potential regions that might contain objects) which can be fed to actual model
    # As it is a pre-processing step,it is implemented on cpu
    gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    gs.setBaseImage(img)
    gs.switchToSelectiveSearchFast()
    rects = gs.process()
    rects[:, 2] += rects[:, 0]
    rects[:, 3] += rects[:, 1]
    print("Suggested number of proposals: %d" % len(rects))

    # Proposals generated by selective search were fed to a model in a loop manner to compute features.
    # [Refer line No.151 in https://github.com/object-detection-algorithm/R-CNN/blob/master/py/car_detector.py]
    for idx, rect in enumerate(rects):

        xmin, ymin, xmax, ymax = rect
        rect_img = img[ymin:ymax, xmin:xmax]

        rect_transform = transform(rect_img)

        inputs = [rect_transform.unsqueeze(0)]

        # Record Forge Property
        module_name = forge_property_recorder.record_model_properties(
            framework=Framework.PYTORCH,
            model="rcnn",
            suffix=f"rect_{idx}",
            source=Source.TORCHVISION,
            task=Task.OBJECT_DETECTION,
        )

        # Forge compile framework model
        compiled_model = forge.compile(
            framework_model,
            sample_inputs=inputs,
            module_name=module_name,
            forge_property_handler=forge_property_recorder,
        )

        verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)

        break  # As generated proposals will be around 2000, halt inference after getting result from single proposal.
