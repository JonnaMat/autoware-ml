# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Plain vision transformer backbone with a simple feature pyramid.

A plain vision transformer keeps one resolution, the grid of its patches. Following ViTDet
("Exploring Plain Vision Transformer Backbones for Object Detection",
https://arxiv.org/abs/2203.16527), most blocks attend within non-overlapping windows of that
grid, a few attend over the whole image, and the feature maps at strides 8, 16 and 32 are
built from the last token map alone. The transformer itself and its pretrained weights come
from timm (https://github.com/huggingface/pytorch-image-models, Apache-2.0).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from jaxtyping import Float
import timm
import torch
import torch.nn as nn

from autoware_ml.models.common.layers.csp import BN_EPS, BN_MOMENTUM, ConvBNAct, build_activation


@contextmanager
def explicit_attention_in_onnx_export(module: nn.Module) -> Iterator[None]:
    """Run the attention layers of a timm network in their explicit form during an ONNX export.

    The fused attention of PyTorch exports its scale as arithmetic on the shape of the tokens.
    The explicit form exports the scaled queries, a matrix product and a softmax, with the
    scale as a constant and applied before the product, which keeps the product small in
    half precision. Outside an export the context changes nothing.

    Args:
        module: Network whose attention layers carry the ``fused_attn`` switch of timm.

    Yields:
        Nothing, the attention layers are switched while the context is open.
    """
    fused = (
        [layer for layer in module.modules() if getattr(layer, "fused_attn", False)]
        if torch.onnx.is_in_onnx_export()
        else []
    )
    for layer in fused:
        layer.fused_attn = False
    try:
        yield
    finally:
        for layer in fused:
            layer.fused_attn = True


class SimpleFeaturePyramid(nn.Module):
    """Feature maps at strides 8, 16 and 32 built from one stride-16 token map.

    The stride-8 map comes from a transposed convolution, the stride-32 map from a max
    pooling. Every level then gets a 1x1 convolution to its output width and a 3x3
    convolution, both with batch normalization and activation as the blocks of the neck.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Sequence[int] = (128, 256, 512),
        activation: str = "silu",
    ) -> None:
        """Initialize the pyramid.

        Args:
            in_channels: Channels of the token map.
            out_channels: Channels of the maps at strides 8, 16 and 32.
            activation: Activation name of every block.

        Raises:
            ValueError: If ``out_channels`` does not name exactly three levels.
        """
        super().__init__()
        if len(out_channels) != 3:
            raise ValueError(f"The pyramid has three levels, got {len(out_channels)} widths.")
        c3, c4, c5 = (int(channels) for channels in out_channels)
        self.out_channels = (c3, c4, c5)
        up_channels = in_channels // 2

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, up_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(up_channels, eps=BN_EPS, momentum=BN_MOMENTUM),
            build_activation(activation),
        )
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.level3 = self._output_block(up_channels, c3, activation)
        self.level4 = self._output_block(in_channels, c4, activation)
        self.level5 = self._output_block(in_channels, c5, activation)

    @staticmethod
    def _output_block(in_channels: int, out_channels: int, activation: str) -> nn.Sequential:
        """Build the 1x1 and 3x3 convolutions of one level."""
        return nn.Sequential(
            ConvBNAct(in_channels, out_channels, 1, 1, activation=activation),
            ConvBNAct(out_channels, out_channels, 3, 1, activation=activation),
        )

    def forward(
        self, x: Float[torch.Tensor, "batch_size in_channels height width"]
    ) -> tuple[Float[torch.Tensor, "batch_size channels level_height level_width"], ...]:
        """Build the three feature maps.

        Args:
            x: Token map at stride 16, its height and width even.

        Returns:
            The feature maps at strides 8, 16 and 32.
        """
        return self.level3(self.up(x)), self.level4(x), self.level5(self.down(x))


class ViTSimplePyramid(nn.Module):
    """Plain vision transformer of timm with windowed attention and a simple feature pyramid.

    The transformer is built for one input size: its position embedding is resampled to the
    patch grid of that size once, when the pretrained weights are loaded, and is a constant of
    the network from then on. A patch size other than the pretrained one resamples the kernel
    of the patch embedding the same way, which keeps the input size of the network: DINOv2
    weights of patch 14 run at patch 16 on a 960x960 input and give a 60x60 grid at stride 16.

    The blocks outside ``global_block_indices`` attend within windows of ``window_size`` by
    ``window_size`` tokens. The class and register tokens are copied into every window and
    averaged over the windows again when the next block attends globally.
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        image_size: Sequence[int] = (960, 960),
        patch_size: int = 16,
        pretrained: bool = True,
        window_size: int | None = None,
        global_block_indices: Sequence[int] = (2, 5, 8, 11),
        out_channels: Sequence[int] = (128, 256, 512),
        drop_path_rate: float = 0.0,
        activation: str = "silu",
    ) -> None:
        """Initialize the backbone.

        Args:
            model_name: Name of a plain vision transformer of timm with its pretrained tag.
            image_size: ``[height, width]`` of the input, both divisible by twice the patch
              size.
            patch_size: Patch size the network runs at, the stride of its token map.
            pretrained: Whether to download and load the pretrained weights of timm. Not
              needed when the weights come from a checkpoint of the detector.
            window_size: Side of the attention windows in tokens, a divisor of the patch
              grid. None for global attention in every block.
            global_block_indices: Blocks that attend over the whole image when the others
              attend within windows.
            out_channels: Channels of the returned maps at strides 8, 16 and 32.
            drop_path_rate: Stochastic depth rate of the transformer blocks.
            activation: Activation name of the pyramid.

        Raises:
            ValueError: If the input size, the window size or the global blocks do not fit
              the network.
        """
        super().__init__()
        height, width = (int(size) for size in image_size)
        if height % (2 * patch_size) or width % (2 * patch_size):
            raise ValueError(
                f"image_size must be divisible by twice the patch size {patch_size}, "
                f"got {[height, width]}."
            )
        self.image_size = (height, width)
        self.patch_size = int(patch_size)
        self.grid_size = (height // patch_size, width // patch_size)

        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=self.image_size,
            patch_size=self.patch_size,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        self.num_prefix_tokens = int(self.vit.num_prefix_tokens)
        self.embed_dim = int(self.vit.embed_dim)
        num_blocks = len(self.vit.blocks)

        if window_size is None:
            self.window_size = None
            self.global_block_indices = tuple(range(num_blocks))
        else:
            self.window_size = int(window_size)
            if (
                self.window_size <= 0
                or self.grid_size[0] % self.window_size
                or self.grid_size[1] % self.window_size
            ):
                raise ValueError(
                    f"window_size must divide the patch grid {self.grid_size}, got {window_size}."
                )
            indices = tuple(sorted({int(index) for index in global_block_indices}))
            if any(not 0 <= index < num_blocks for index in indices):
                raise ValueError(
                    f"global_block_indices must select blocks in [0, {num_blocks - 1}], "
                    f"got {indices}."
                )
            self.global_block_indices = indices

        self.pyramid = SimpleFeaturePyramid(self.embed_dim, out_channels, activation=activation)
        self.out_channels = self.pyramid.out_channels
        self.out_strides = (self.patch_size // 2, self.patch_size, self.patch_size * 2)

    def _to_windows(
        self, x: Float[torch.Tensor, "batch_size num_tokens channels"]
    ) -> Float[torch.Tensor, "num_windows window_tokens channels"]:
        """Split the patch tokens into windows and copy the prefix tokens into each."""
        size = self.window_size
        rows, columns = self.grid_size[0] // size, self.grid_size[1] // size
        prefix, patches = x[:, : self.num_prefix_tokens], x[:, self.num_prefix_tokens :]
        patches = patches.reshape(-1, rows, size, columns, size, self.embed_dim)
        patches = patches.permute(0, 1, 3, 2, 4, 5).reshape(-1, size * size, self.embed_dim)
        if self.num_prefix_tokens == 0:
            return patches
        # One copy per window. A concatenation, unlike an expansion, exports without shape
        # arithmetic.
        prefix = torch.cat([prefix] * (rows * columns), dim=1)
        prefix = prefix.reshape(-1, self.num_prefix_tokens, self.embed_dim)
        return torch.cat([prefix, patches], dim=1)

    def _from_windows(
        self, x: Float[torch.Tensor, "num_windows window_tokens channels"]
    ) -> Float[torch.Tensor, "batch_size num_tokens channels"]:
        """Join the windows into one token sequence and average their prefix tokens."""
        size = self.window_size
        rows, columns = self.grid_size[0] // size, self.grid_size[1] // size
        prefix, patches = x[:, : self.num_prefix_tokens], x[:, self.num_prefix_tokens :]
        patches = patches.reshape(-1, rows, columns, size, size, self.embed_dim)
        patches = patches.permute(0, 1, 3, 2, 4, 5)
        patches = patches.reshape(-1, self.grid_size[0] * self.grid_size[1], self.embed_dim)
        if self.num_prefix_tokens == 0:
            return patches
        prefix = prefix.reshape(-1, rows * columns, self.num_prefix_tokens, self.embed_dim)
        return torch.cat([prefix.mean(dim=1), patches], dim=1)

    def _embed(
        self, images: Float[torch.Tensor, "batch_size 3 height width"]
    ) -> Float[torch.Tensor, "batch_size num_tokens channels"]:
        """Embed the patches, prepend the prefix tokens and add the position embedding.

        The position embedding has the size of the patch grid, so it is added as it is. A
        single image, the deployed case, takes the prefix tokens without an expansion over
        the batch, which would export as arithmetic on the shape of the input.

        Raises:
            ValueError: If the images are not of the input size of the network.
        """
        vit = self.vit
        if tuple(images.shape[-2:]) != self.image_size:
            raise ValueError(
                f"The backbone is built for images of {self.image_size}, "
                f"got {tuple(images.shape[-2:])}."
            )
        # The patch grid is known, so the tokens are laid out without reading their shape.
        x = vit.patch_embed.proj(images)
        x = x.reshape(-1, self.embed_dim, self.grid_size[0] * self.grid_size[1]).transpose(1, 2)
        x = vit.patch_embed.norm(x)
        if vit.no_embed_class:
            x = x + vit.pos_embed
        prefix = [token for token in (vit.cls_token, vit.reg_token) if token is not None]
        if prefix:
            tokens = torch.cat(prefix, dim=1)
            if x.shape[0] != 1:
                tokens = tokens.expand(x.shape[0], -1, -1)
            x = torch.cat([tokens, x], dim=1)
        if not vit.no_embed_class:
            x = x + vit.pos_embed
        return vit.pos_drop(x)

    def forward_tokens(
        self, images: Float[torch.Tensor, "batch_size 3 height width"]
    ) -> Float[torch.Tensor, "batch_size channels grid_height grid_width"]:
        """Compute the normalized patch tokens of the last block as a stride-16 map.

        Args:
            images: Images of the input size of the network.

        Returns:
            The token map, without the class and register tokens.
        """
        vit = self.vit
        x = self._embed(images)
        x = vit.patch_drop(x)
        x = vit.norm_pre(x)

        with explicit_attention_in_onnx_export(vit):
            windowed = False
            for index, block in enumerate(vit.blocks):
                attend_globally = index in self.global_block_indices
                if windowed and attend_globally:
                    x = self._from_windows(x)
                elif not windowed and not attend_globally:
                    x = self._to_windows(x)
                windowed = not attend_globally
                x = block(x)
            if windowed:
                x = self._from_windows(x)

        x = vit.norm(x)[:, self.num_prefix_tokens :]
        return x.transpose(1, 2).reshape(-1, self.embed_dim, *self.grid_size)

    def forward(
        self, images: Float[torch.Tensor, "batch_size 3 height width"]
    ) -> tuple[Float[torch.Tensor, "batch_size channels level_height level_width"], ...]:
        """Extract the feature maps at strides 8, 16 and 32.

        Args:
            images: Images of the input size of the network, normalized as its weights expect.

        Returns:
            The three feature maps, from the shallowest to the deepest.
        """
        return self.pyramid(self.forward_tokens(images))

    def lr_layer_ids(self) -> tuple[dict[str, int], int]:
        """Assign the pretrained parameters to the layers of a layer-wise learning rate decay.

        The embeddings are layer 0, block ``i`` is layer ``i + 1`` and the final normalization
        is layer ``num_blocks + 1``. The pyramid is new and belongs to no layer.

        Returns:
            The layer of every parameter of the transformer by its name within the backbone,
            and the highest layer.
        """
        num_blocks = len(self.vit.blocks)
        layer_ids = {}
        for name, _ in self.vit.named_parameters():
            if name.startswith("blocks."):
                layer_ids[f"vit.{name}"] = int(name.split(".")[1]) + 1
            elif name.startswith("norm."):
                layer_ids[f"vit.{name}"] = num_blocks + 1
            else:
                layer_ids[f"vit.{name}"] = 0
        return layer_ids, num_blocks + 1

    def no_weight_decay(self) -> set[str]:
        """Name the parameters weight decay skips whatever their shape: the embeddings.

        Returns:
            The parameter names within the backbone.
        """
        return {f"vit.{name}" for name in self.vit.no_weight_decay()}
