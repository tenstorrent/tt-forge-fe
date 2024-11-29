# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
## Source
## https://github.com/stigma0617/VoVNet.pytorch/blob/039d9dcbd07de73fb5b8cc14bdde483a3807225c/models_vovnet/vovnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# from torch.hub import load_state_dict_from_url
from torch.utils.model_zoo import load_url as load_state_dict_from_url

from transformers import AutoImageProcessor
from PIL import Image
import requests

__all__ = ["VoVNet", "vovnet27_slim", "vovnet39", "vovnet57"]


model_urls = {
    "vovnet39": "https://dl.dropbox.com/s/1lnzsgnixd8gjra/vovnet39_torchvision.pth?dl=1",
    "vovnet57": "https://dl.dropbox.com/s/6bfu9gstbwfw31m/vovnet57_torchvision.pth?dl=1",
}


def conv3x3(in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [
        (
            "{}_{}/conv".format(module_name, postfix),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        ),
        ("{}_{}/norm".format(module_name, postfix), nn.BatchNorm2d(out_channels)),
        ("{}_{}/relu".format(module_name, postfix), nn.ReLU(inplace=True)),
    ]


def conv1x1(in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=1, padding=0):
    """1x1 convolution"""
    return [
        (
            "{}_{}/conv".format(module_name, postfix),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        ),
        ("{}_{}/norm".format(module_name, postfix), nn.BatchNorm2d(out_channels)),
        ("{}_{}/relu".format(module_name, postfix), nn.ReLU(inplace=True)),
    ]


class _OSA_module(nn.Module):
    def __init__(self, in_ch, stage_ch, concat_ch, layer_per_block, module_name, identity=False):
        super(_OSA_module, self).__init__()

        self.identity = identity
        self.layers = nn.ModuleList()
        in_channel = in_ch
        for i in range(layer_per_block):
            self.layers.append(nn.Sequential(OrderedDict(conv3x3(in_channel, stage_ch, module_name, i))))
            in_channel = stage_ch

        # feature aggregation
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(OrderedDict(conv1x1(in_channel, concat_ch, module_name, "concat")))

    def forward(self, x):
        identity_feat = x
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        xt = self.concat(x)

        if self.identity:
            xt = xt + identity_feat

        return xt


class _OSA_stage(nn.Sequential):
    def __init__(self, in_ch, stage_ch, concat_ch, block_per_stage, layer_per_block, stage_num):
        super(_OSA_stage, self).__init__()

        if not stage_num == 2:
            self.add_module("Pooling", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        module_name = f"OSA{stage_num}_1"
        self.add_module(module_name, _OSA_module(in_ch, stage_ch, concat_ch, layer_per_block, module_name))
        for i in range(block_per_stage - 1):
            module_name = f"OSA{stage_num}_{i+2}"
            self.add_module(
                module_name, _OSA_module(concat_ch, stage_ch, concat_ch, layer_per_block, module_name, identity=True)
            )


class VoVNet(nn.Module):
    def __init__(self, config_stage_ch, config_concat_ch, block_per_stage, layer_per_block, num_classes=1000):
        super(VoVNet, self).__init__()

        # Stem module
        stem = conv3x3(3, 64, "stem", "1", 2)
        stem += conv3x3(64, 64, "stem", "2", 1)
        stem += conv3x3(64, 128, "stem", "3", 2)
        self.add_module("stem", nn.Sequential(OrderedDict(stem)))

        stem_out_ch = [128]
        in_ch_list = stem_out_ch + config_concat_ch[:-1]
        self.stage_names = []
        for i in range(4):  # num_stages
            name = "stage%d" % (i + 2)
            self.stage_names.append(name)
            self.add_module(
                name,
                _OSA_stage(
                    in_ch_list[i], config_stage_ch[i], config_concat_ch[i], block_per_stage[i], layer_per_block, i + 2
                ),
            )

        self.classifier = nn.Linear(config_concat_ch[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        for name in self.stage_names:
            x = getattr(self, name)(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        return x


def _vovnet(arch, config_stage_ch, config_concat_ch, block_per_stage, layer_per_block, pretrained, progress, **kwargs):
    model = VoVNet(config_stage_ch, config_concat_ch, block_per_stage, layer_per_block, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], map_location=torch.device("cpu"), progress=progress)
        # print(state_dict.keys())
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k[7:]
            new_state_dict[k] = v
        # print(new_state_dict.keys())
        model.load_state_dict(new_state_dict)
    return model


def vovnet57(pretrained=False, progress=True, **kwargs):
    r"""Constructs a VoVNet-57 model as described in
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vovnet(
        "vovnet57", [128, 160, 192, 224], [256, 512, 768, 1024], [1, 1, 4, 3], 5, pretrained, progress, **kwargs
    )


def vovnet39(pretrained=False, progress=True, **kwargs):
    r"""Constructs a VoVNet-39 model as described in
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vovnet(
        "vovnet39", [128, 160, 192, 224], [256, 512, 768, 1024], [1, 1, 2, 2], 5, pretrained, progress, **kwargs
    )


def vovnet27_slim(pretrained=False, progress=True, **kwargs):
    r"""Constructs a VoVNet-39 model as described in
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vovnet(
        "vovnet27_slim", [64, 80, 96, 112], [128, 256, 384, 512], [1, 1, 1, 1], 5, pretrained, progress, **kwargs
    )
