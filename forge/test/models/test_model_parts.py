# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

import forge
from forge.verify.verify import verify
import math
import onnx
from torchvision.models.detection import _utils as det_utils
from test.models.pytorch.vision.vision_utils.utils import load_vision_model_and_input


@pytest.mark.skip_model_analysis
@pytest.mark.xfail(
    reason="AssertionError: Data mismatch on output 0 between framework and Forge codegen, PCC got=0.4923030518607919"
)  # https://github.com/tenstorrent/tt-forge-fe/issues/1793
@pytest.mark.push
def test_inplace_updation():
    class Inplace_updation(nn.Module):
        def __init__(self):
            super().__init__()
            self.shift_size = 4
            self.window_size = 8

        def forward(self, x):

            img_mask = torch.zeros((1, 64, 64, 1), dtype=torch.float32)
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            return img_mask + x

    inputs = [torch.randn((1, 64, 64, 1), dtype=torch.float32)]
    model = Inplace_updation()
    model.eval()

    compiled_model = forge.compile(
        model,
        sample_inputs=inputs,
    )
    verify(inputs, model, compiled_model)


@pytest.mark.parametrize(
    "input_shape, clamp_min, clamp_max, dtype",
    [
        ((64, 3, 64, 64), None, math.log(100.0), torch.float32),
        ((11, 30, 11, 11), math.log(101.0), None, torch.float32),
        ((27, 16, 27, 27), -math.log(103.0), None, torch.float32),
        ((45, 2, 45, 45), None, 5.0, torch.float32),
        ((3, 21, 3, 3), None, math.log(50.0), torch.float32),
        ((12, 6, 12, 12), -103.0, None, torch.float32),
        ((18, 11, 18, 18), -50, None, torch.int32),
        ((8, 1, 8, 8), None, 876, torch.int32),
    ],
)
@pytest.mark.skip_model_analysis
@pytest.mark.push
def test_clamp(input_shape, clamp_min, clamp_max, dtype):
    class Clamp(nn.Module):
        def __init__(self):
            super().__init__()
            log_value = torch.log(10 * torch.ones(input_shape[1], 1, 1))
            self.logit_scale = nn.Parameter(log_value)

        def forward(self, attn):
            clamped = torch.clamp(self.logit_scale, min=clamp_min, max=clamp_max)
            logit_scale = clamped.exp()
            return attn * logit_scale

    model = Clamp()
    model.eval()

    if dtype == torch.float32:
        attn = torch.randn(input_shape)
    elif dtype == torch.int32:
        attn = torch.randint(low=torch.iinfo(dtype).min, high=torch.iinfo(dtype).max, size=input_shape, dtype=dtype)

    inputs = [attn]

    # Export to ONNX
    torch.onnx.export(model, (inputs[0],), "temp.onnx", opset_version=17)

    # Load and check ONNX
    onnx_model = onnx.load("temp.onnx")
    onnx.checker.check_model(onnx_model)

    compiled_model = forge.compile(onnx_model, inputs)
    verify(inputs, model, compiled_model)


@pytest.mark.skip_model_analysis
@pytest.mark.push
def test_rotary_pos_emb():
    class RotaryPosEmb(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, q, cos, sin):
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

            rotary_dim = cos.shape[-1]
            q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]

            q_embed = torch.cat([q_pass, (q_rot * cos) + (self.rotate_half(q_rot) * sin)], dim=-1)
            return q_embed

        def rotate_half(self, x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

    framework_model = RotaryPosEmb()
    framework_model.eval()

    query = torch.rand((1, 32, 256, 96))
    cos_emb = torch.rand([1, 256, 96])
    sin_emb = torch.rand([1, 256, 96])
    inputs = [query, cos_emb, sin_emb]

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    verify(inputs, framework_model, compiled_model)


@pytest.mark.skip_model_analysis
@pytest.mark.push
@pytest.mark.parametrize(
    "dim",
    [-4, -3, -2, -1, 0, 1, 2, 3],
)
def test_remove_concat_pass(dim):
    class SliceConcat(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, q):
            sl = [slice(None)] * len(q.shape)
            sl[self.dim] = slice(q.shape[self.dim], None)
            q_pass = q[tuple(sl)]
            q_embed = torch.cat([q_pass, (q * q)], dim=self.dim)
            return q_embed

    model = SliceConcat(dim)

    query = torch.rand((1, 32, 256, 96))
    inputs = [query]

    compiled_model = forge.compile(model, sample_inputs=inputs)
    verify(inputs, model, compiled_model)


@pytest.mark.parametrize(
    "input_shape, flip_dim",
    [
        ((1, 1, 1024, 72), 0),
        ((1, 1, 104, 72), 1),
        ((1, 12, 1, 92), 2),
        ((1, 33, 11, 1), 3),
        ((1, 1, 17, 16), 1),
    ],
)
def test_flip(input_shape, flip_dim):
    class Flip(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, x):
            return x.flip(dims=(self.dim,))

    torch_model = Flip(dim=flip_dim)
    inputs = [torch.rand(*input_shape)]

    # Export model to ONNX
    onnx_path = "flip.onnx"
    torch.onnx.export(torch_model, (inputs[0],), onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule("sanity", onnx_model)

    # Compile model
    compiled_model = forge.compile(onnx_model, inputs)

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
    )


@pytest.mark.skip_model_analysis
@pytest.mark.push
def test_stack_and_reshape_onnx():
    class stack_reshape(nn.Module):
        def __init__(self):
            super().__init__()
            self.boxes_shape = (3234, 4)

        def forward(self, boxes_x, boxes_y):
            clipped_boxes = torch.stack((boxes_x, boxes_y), dim=2)
            return clipped_boxes.reshape(self.boxes_shape)

    torch_model = stack_reshape()
    x = torch.randn(3234, 2)
    y = torch.randn(3234, 2)
    inputs = [x, y]

    onnx_path = "stack_reshape.onnx"
    torch.onnx.export(
        torch_model,
        (inputs[0], inputs[1]),
        onnx_path,
        opset_version=17,
    )

    module_name = "stack_reshape"
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=inputs,
        module_name=module_name,
    )

    verify(inputs, framework_model, compiled_model)


variants_with_weights = {
    "ssdlite320_mobilenet_v3_large": "SSDLite320_MobileNet_V3_Large_Weights",
}


@pytest.mark.nightly
@pytest.mark.skip_model_analysis
@pytest.mark.parametrize("variant", variants_with_weights.keys())
def test_ssdlite320_mobilenet_v3_large_problematic_block(variant):

    pytest.xfail(reason="Fatal Python error: Segmentation fault")

    class postprocess_detections(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.image_sizes = [(320, 320)]

        def forward(self, input1, input2, input3):

            head_outputs = {"bbox_regression": input1, "cls_logits": input2}
            anchors = [input3]

            op = self.model.postprocess_detections(head_outputs, anchors, self.image_sizes)
            return op

    # Load model
    weight_name = variants_with_weights[variant]
    framework_model, _ = load_vision_model_and_input(variant, "detection", weight_name)
    framework_model = postprocess_detections(framework_model)

    bbox_regression = torch.randn(1, 3234, 4)
    cls_logits = torch.randn(1, 3234, 91)
    anchors = torch.randn(3234, 4)
    inputs = [bbox_regression, cls_logits, anchors]

    onnx_path = "problematic_block.onnx"
    torch.onnx.export(
        framework_model,
        (inputs[0], inputs[1], inputs[2]),
        onnx_path,
        opset_version=17,
    )

    module_name = "problematic_block"
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=inputs,
        module_name=module_name,
    )
    verify(inputs, framework_model, compiled_model)


@pytest.mark.skip_model_analysis
def test_gather_to_take_onnx():
    class take(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, boxes, score):

            keep_idxs = score > 0.001
            score = score[keep_idxs]
            box = boxes[keep_idxs]
            _, idxs = score.topk(300)
            box = box[idxs]

            return box

    torch_model = take()
    boxes = torch.rand(3234, 4)
    scores = torch.rand(3234)
    inputs = [boxes, scores]

    onnx_path = "take.onnx"
    torch.onnx.export(
        torch_model,
        (inputs[0], inputs[1]),
        onnx_path,
        opset_version=17,
        verbose=True,
    )

    module_name = "gather_to_take_onnx"
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    framework_model = forge.OnnxModule(module_name, onnx_model)
    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=inputs,
        module_name=module_name,
    )

    verify(inputs, framework_model, compiled_model)


def test_concat_block():
    class concat(nn.Module):
        def __init__(self):
            super().__init__()
            self.score_thresh = 0.001
            self.topk_candidates = 300

        def forward(self, score):

            keep_idxs = score > self.score_thresh
            score = score[keep_idxs]
            num_topk = det_utils._topk_min(score, self.topk_candidates, 0)

            return num_topk

    torch_model = concat()
    scores = torch.rand(3234)
    inputs = [scores]

    onnx_path = "concat.onnx"
    torch.onnx.export(
        torch_model,
        (inputs[0]),
        onnx_path,
        opset_version=17,
        verbose=True,
    )

    module_name = "concat"
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=inputs,
        module_name=module_name,
    )

    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.skip_model_analysis
@pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-forge-fe/issues/2899")
@pytest.mark.parametrize(
    "tensor_size,max_length",
    [
        (24, 160),
        (10, 50),
        (32, 100),
        (16, 200),
        (8, 5),
    ],
)
def test_index_put_speecht5_tts(tensor_size, max_length):
    class index_put(nn.Module):
        def __init__(self, max_length):
            super().__init__()
            self.max_length = max_length

        def forward(self, pos_seq):

            pos_seq[pos_seq < -self.max_length] = -self.max_length
            return pos_seq

    pos_seq = torch.arange(tensor_size).unsqueeze(1) - torch.arange(tensor_size).unsqueeze(0)

    inputs = [pos_seq]
    model = index_put(max_length)
    model.eval()

    compiled_model = forge.compile(
        model,
        sample_inputs=inputs,
    )
    verify(inputs, model, compiled_model)


@pytest.mark.xfail
@pytest.mark.nightly
@pytest.mark.skip_model_analysis
@pytest.mark.parametrize(
    "tensor_size,num_heads,head_dim,max_length",
    [
        (24, 12, 64, 160),
        (16, 8, 32, 100),
        (32, 16, 64, 200),
        (8, 4, 16, 50),
        (12, 6, 128, 80),
    ],
)
def test_view_speecht5_tts(tensor_size, num_heads, head_dim, max_length):
    class view(nn.Module):
        def __init__(self, num_heads, head_dim, max_length):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.dim = head_dim
            self.max_length = max_length
            self.pe_k = torch.nn.Embedding(2 * max_length, head_dim)

        def forward(self, pos_seq, reshape_q):
            pos_seq[pos_seq < -self.max_length] = -self.max_length
            pos_seq[pos_seq >= self.max_length] = self.max_length - 1
            pos_seq = pos_seq + self.max_length
            position_bias = self.pe_k(pos_seq)
            rel_pos_bias = torch.matmul(reshape_q, position_bias.transpose(-2, -1))
            rel_pos_bias = rel_pos_bias.transpose(0, 1)
            rel_pos_bias = rel_pos_bias.view(1 * self.num_heads, position_bias.size(0), position_bias.size(1))
            return rel_pos_bias

    x = torch.arange(tensor_size).unsqueeze(1) - torch.arange(tensor_size).unsqueeze(0)
    y = torch.randn((tensor_size, num_heads, head_dim), dtype=torch.float32)

    inputs = [x, y]
    model = view(num_heads, head_dim, max_length)
    model.eval()

    compiled_model = forge.compile(
        model,
        sample_inputs=inputs,
    )
    verify(inputs, model, compiled_model)
