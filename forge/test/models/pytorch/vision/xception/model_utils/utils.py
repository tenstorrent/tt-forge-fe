# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import requests
import torch

url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"


def post_processing(output, top_k=5):

    probabilities = torch.nn.functional.softmax(output[0][0], dim=0)
    r = requests.get(url, allow_redirects=True)
    categories = [s.strip() for s in r.content.decode("utf-8").splitlines()]
    topk_prob, topk_catid = torch.topk(probabilities, top_k)
    for i in range(topk_prob.size(0)):
        print(categories[topk_catid[i]], topk_prob[i].item())
