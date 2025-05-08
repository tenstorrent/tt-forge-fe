import pytest
import torch
from torch import nn

from transformers import AutoModelForCausalLM, AutoProcessor
from transformers import CLIPVisionConfig, CLIPVisionModel, PretrainedConfig
from test.utils import download_model

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.pytorch.multimodal.phi3.utils.utils import load_input

variants = ["microsoft/Phi-3.5-vision-instruct"]
from loguru import logger

CLIP_VIT_LARGE_PATCH14_336_CONFIG = CLIPVisionConfig(
  attention_dropout=0.0,
  dropout=0.0,
  hidden_act="quick_gelu",
  hidden_size=1024,
  image_size=336,
  initializer_factor=1.0,
  initializer_range=0.02,
  intermediate_size=4096,
  layer_norm_eps=1e-05,
  num_attention_heads=16,
  num_channels=3,
  num_hidden_layers=24,
  patch_size=14,
  projection_dim=768
)
MAX_INPUT_ID = int(1e9)

@pytest.mark.parametrize("variant", variants)
def test_embedding_sanity(forge_property_recorder, variant):
    class Phi3ImageEmbedding(torch.nn.Module):
        """Phi3 Image embedding."""

        def __init__(self, wte=None) -> None:
            super().__init__()

            # n_embed or hidden_size
            hidden_size = 3072
            embd_drop = 0.0
            self.drop = nn.Dropout(embd_drop)
            self.wte = nn.Embedding(32064, 3072, 32000)

            clip_config = CLIP_VIT_LARGE_PATCH14_336_CONFIG
            self.img_processor = CLIPVisionModel(clip_config)
            image_dim_out = 1024
            self.num_img_tokens = 144


            self.image_dim_out = image_dim_out
            self.img_sizes = None

            # global_gn and sub_gn for hd transform, serves as line separator
            self.use_hd_transform = True
            self.with_learnable_separator = True
            self.hd_transform_order = 'sub_glb'
            # with_hd_transform and with_learnable_separator should have same value
            assert self.use_hd_transform == self.with_learnable_separator, 'use_hd_transform and with_learnable_separator should have same value'
            if self.with_learnable_separator:
                assert self.use_hd_transform, 'learnable separator is only for hd transform'
                # 1024 * 4, merge spatial to channel dimension
                self.glb_GN = nn.Parameter(torch.zeros([1, 1, self.image_dim_out * 4]))
                self.sub_GN = nn.Parameter(torch.zeros([1, 1, 1, self.image_dim_out * 4]))
                logger.info(f'learnable separator enabled for hd transform, hd_transform_order = {self.hd_transform_order}')

            projection_cls = 'mlp'
            if projection_cls == 'linear':
                self.img_projection = nn.Linear(image_dim_out, hidden_size)
            elif projection_cls == 'mlp' and self.use_hd_transform:
                dim_projection = hidden_size
                depth = 2
                layers = [nn.Linear(image_dim_out * 4, dim_projection)]
                for _ in range(1, depth):
                    layers.extend([nn.GELU(),
                                    nn.Linear(dim_projection, dim_projection)])
                self.img_projection = nn.Sequential(*layers)
            elif projection_cls == 'mlp':
                dim_projection = hidden_size
                depth = 2
                layers = [nn.Linear(image_dim_out, dim_projection)]
                for _ in range(1, depth):
                    layers.extend([nn.GELU(),
                                    nn.Linear(dim_projection, dim_projection)])
                self.img_projection = nn.Sequential(*layers)
            else:
                raise NotImplementedError(f'projection_cls = {projection_cls}, not implemented')

            self.vocab_size = 32064
            self.img_features = None

            self.layer_idx = -2
            self.type_feature = 'patch'


        def set_img_features(self, img_features: torch.FloatTensor) -> None:
            self.img_features = img_features

        def set_img_sizes(self, img_sizes: torch.LongTensor) -> None:
            self.img_sizes = img_sizes

        def get_img_features(self, img_embeds: torch.FloatTensor) -> torch.FloatTensor:
            LAYER_IDX = self.layer_idx
            TYPE_FEATURE = self.type_feature

            img_processor_output = self.img_processor(img_embeds, output_hidden_states=True)
            img_feature = img_processor_output.hidden_states[LAYER_IDX]

            if TYPE_FEATURE == "patch":
                patch_feature = img_feature[:, 1:]
                return patch_feature

            raise NotImplementedError

        def forward(
            self, input_ids: torch.LongTensor, pixel_values: torch.FloatTensor, image_sizes=None
        ) -> torch.FloatTensor:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])

            # positions for image tokens
            positions = torch.nonzero((input_ids < 0) & (input_ids > -MAX_INPUT_ID), as_tuple=True)
            has_image = len(positions[0].tolist()) > 0
            input_ids = input_ids.clamp_min(0).clamp_max(self.vocab_size).detach()
            hidden_states = self.wte(input_ids)
            if has_image:
                assert self.use_hd_transform
                num_images, num_crops, c, h, w = pixel_values.shape
                assert c == 3 and h == w == 336
                img_features = self.get_img_features(pixel_values.flatten(0, 1)).reshape(
                    num_images, num_crops, -1, self.image_dim_out
                )
                image_features_proj = self.hd_feature_transform(img_features, image_sizes)
                hidden_states = hidden_states.index_put(
                    positions, image_features_proj, accumulate=False
                )
            return hidden_states
        
        def hd_feature_transform(self, image_features, image_sizes):
            """
            image_features: (num_images, num_crops+1, 24*24, 1024)
            """
            assert (
                self.hd_transform_order == 'sub_glb'
            ), f'hd_transform_order `{self.hd_transform_order}` not implemented'
            if isinstance(self.img_projection, nn.Sequential):
                target_device = self.img_projection[0].bias.device
                target_dtype = self.img_projection[0].bias.dtype
            else:  # It's a single nn.Linear layer
                target_device = self.img_projection.bias.device
                target_dtype = self.img_projection.bias.dtype

            global_image_features = image_features[:, 0]  # (num_images, 24*24, 1024)
            # global feature can be viewed as a special HD case with num_crops 1x1
            global_image_features_hd = self.reshape_hd_patches_2x2merge(global_image_features, 1, 1)
            global_image_features_hd_newline = self.add_image_newline(global_image_features_hd)

            all_image_embeddings = []
            # need a for loop to process each image because of different image sizes
            # (patch arrangement is different for each image)
            for i, img_size in enumerate(image_sizes):
                logger.info(f" i = {i}")
                h, w = img_size
                h_crop = h // 336
                w_crop = w // 336
                num_crops = h_crop * w_crop

                # NOTE: real num_crops is padded
                # (num_crops, 24*24, 1024)
                sub_image_features = image_features[i, 1 : 1 + num_crops]
                sub_image_features_hd = self.reshape_hd_patches_2x2merge(
                    sub_image_features, h_crop, w_crop
                )
                sub_image_features_hd_newline = self.add_image_newline(sub_image_features_hd)

                # [sub features, separator, global features]
                all_image_embeddings.extend(
                    [
                        sub_image_features_hd_newline.squeeze(0),  # (h_crop*12*(w_crop*12+1), 4096)
                        self.glb_GN.squeeze(0),
                        global_image_features_hd_newline[i],
                    ]
                )
            image_features_proj = self.img_projection(
                torch.cat(all_image_embeddings, dim=0).to(target_device).to(target_dtype)
            )
            return image_features_proj

        def reshape_hd_patches_2x2merge(self, image_features, h_crop, w_crop):
            """
            image_features: (num_images*num_crops, 24*24, 1024)
            output: (num_images, h_crop*12, w_crop*12, 4096), h_crop*w_crop == num_crops
            """
            N, L, C = image_features.shape
            assert L == 24 * 24 and C == 1024 and N % (h_crop * w_crop) == 0
            num_images = N // (h_crop * w_crop)
            H = int(L**0.5)
            image_features_hd = (
                image_features.reshape(N, H, H, C)  # N, 24, 24, 1024
                .reshape(N, H // 2, 2, H // 2, 2, C)  # N, 12, 2, 12, 2, 1024
                .permute(0, 1, 3, 2, 4, 5)  # N, 12, 12, 2, 2, 1024
                .reshape(N, -1, 4 * C)  # N, 144, 4096
                .reshape(
                    num_images, h_crop, w_crop, H // 2, H // 2, -1
                )  # n_img, h_crop, w_crop, 12, 12, 4096
                .permute(0, 1, 3, 2, 4, 5)  # n_img, h_crop, 12, w_crop, 12, 4096
                .reshape(
                    num_images, h_crop * H // 2, w_crop * H // 2, 4 * C
                )  # n_img, h_crop*12, w_crop*12, 4096
            )

            return image_features_hd

        def add_image_newline(self, image_features_hd):
            """
            image_features_hd: (num_images, h_crop*12, w_crop*12, 4096)
            output: (num_images, (h_crop*12) * (w_crop*12+1), 4096)
            """
            num_images, h, w, hid_dim = image_features_hd.shape
            # add the newline token to the HD image feature patches
            newline_embeddings = self.sub_GN.expand(num_images, h, -1, -1)  # (n_img, h, 1, hid_dim)
            image_features_hd_newline = torch.cat(
                [image_features_hd, newline_embeddings], dim=2
            ).reshape(num_images, -1, hid_dim)
            return image_features_hd_newline
    
    # prepare input
    processor = download_model(AutoProcessor.from_pretrained, variant, trust_remote_code=True, num_crops=4)
    inputs = load_input(processor)
    input_ids, attention_mask, pixel_values, image_sizes = inputs
    framework_model = Phi3ImageEmbedding()
    
    inputs_new = [input_ids, pixel_values, image_sizes]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs_new, module_name="embedding_sanity", forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs_new, framework_model, compiled_model, forge_property_handler=forge_property_recorder)