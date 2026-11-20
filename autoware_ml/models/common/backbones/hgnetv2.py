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

"""HGNetV2 backbone.

Ported from pytorch-image-models (``timm/models/hgnet.py`` of
https://github.com/huggingface/pytorch-image-models, Apache-2.0), which adapts PP-HGNetV2 of
PaddleClas (https://github.com/PaddlePaddle/PaddleClas, Apache-2.0). The submodule names follow
timm, so a timm state dict of the same variant loads without a key rename once its
classification head is dropped. The backbone follows the multi-scale tuple interface of the
other image backbones: it returns the feature maps of the selected stages, by default those
at strides 8, 16 and 32.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import logging
from pathlib import Path

from jaxtyping import Float
import torch
import torch.nn as nn
import torch.nn.functional as F

from autoware_ml.models.common.layers.csp import ConvBNAct

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _StageSpec:
    """Layout of one stage."""

    mid_channels: int
    out_channels: int
    num_blocks: int
    downsample: bool
    light_block: bool
    kernel_size: int
    num_layers: int = 3


@dataclass(frozen=True)
class _VariantSpec:
    """Layout of one HGNetV2 variant and its published weight sets."""

    stem_channels: tuple[int, int]
    stages: tuple[_StageSpec, _StageSpec, _StageSpec, _StageSpec]
    weight_urls: dict[str, str]


_HUB_URL = "https://huggingface.co/timm/hgnetv2_{variant}.{tag}/resolve/main/pytorch_model.bin"

# Stage layouts of timm. The weight sets are the timm releases (Apache-2.0): distilled on
# ImageNet-22k, then fine-tuned on ImageNet-1k. They read RGB normalized by the ImageNet
# statistics.
_VARIANTS: dict[str, _VariantSpec] = {
    "b0": _VariantSpec(
        stem_channels=(16, 16),
        stages=(
            _StageSpec(16, 64, 1, False, False, 3),
            _StageSpec(32, 256, 1, True, False, 3),
            _StageSpec(64, 512, 2, True, True, 5),
            _StageSpec(128, 1024, 1, True, True, 5),
        ),
        weight_urls={
            "ssld_stage2_ft_in1k": _HUB_URL.format(variant="b0", tag="ssld_stage2_ft_in1k"),
        },
    ),
    "b1": _VariantSpec(
        stem_channels=(24, 32),
        stages=(
            _StageSpec(32, 64, 1, False, False, 3),
            _StageSpec(48, 256, 1, True, False, 3),
            _StageSpec(96, 512, 2, True, True, 5),
            _StageSpec(192, 1024, 1, True, True, 5),
        ),
        weight_urls={
            "ssld_stage2_ft_in1k": _HUB_URL.format(variant="b1", tag="ssld_stage2_ft_in1k"),
        },
    ),
}


class LearnableAffine(nn.Module):
    """One learnable scale and one learnable offset shared by every element of the input."""

    def __init__(self) -> None:
        """Initialize the block as the identity."""
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: Float[torch.Tensor, "*shape"]) -> Float[torch.Tensor, "*shape"]:
        """Scale and shift the input."""
        return self.scale * x + self.bias


class HGConvBN(nn.Module):
    """Convolution and batch normalization, optionally a ReLU and a learnable affine block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        use_act: bool = True,
        use_lab: bool = True,
    ) -> None:
        """Initialize the block.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            kernel_size: Kernel size, padded by ``(kernel_size - 1) // 2`` on every side.
            stride: Convolution stride.
            groups: Convolution groups.
            use_act: Whether a ReLU follows the normalization.
            use_lab: Whether a learnable affine block follows the ReLU.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU() if use_act else nn.Identity()
        self.lab = LearnableAffine() if use_act and use_lab else nn.Identity()

    def forward(
        self, x: Float[torch.Tensor, "batch_size in_channels height width"]
    ) -> Float[torch.Tensor, "batch_size out_channels out_height out_width"]:
        """Apply the convolution, the normalization, the activation and the affine block."""
        return self.lab(self.act(self.bn(self.conv(x))))


class HGLightConvBN(nn.Module):
    """A 1x1 convolution followed by a depthwise convolution, the cheap layer of deep stages."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_lab: bool = True,
    ) -> None:
        """Initialize the layer.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            kernel_size: Kernel size of the depthwise convolution.
            use_lab: Whether a learnable affine block follows the ReLU.
        """
        super().__init__()
        self.conv1 = HGConvBN(in_channels, out_channels, 1, use_act=False, use_lab=False)
        self.conv2 = HGConvBN(
            out_channels,
            out_channels,
            kernel_size,
            groups=out_channels,
            use_lab=use_lab,
        )

    def forward(
        self, x: Float[torch.Tensor, "batch_size in_channels height width"]
    ) -> Float[torch.Tensor, "batch_size out_channels height width"]:
        """Apply the pointwise and the depthwise convolution."""
        return self.conv2(self.conv1(x))


class HGStem(nn.Module):
    """Stride 4 stem: a strided convolution, a two-branch block and a second strided one."""

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        use_lab: bool = True,
    ) -> None:
        """Initialize the stem.

        Args:
            in_channels: Channels of the input image.
            mid_channels: Channels of the first convolution.
            out_channels: Channels of the stem output.
            use_lab: Whether a learnable affine block follows every ReLU.
        """
        super().__init__()
        self.stem1 = HGConvBN(in_channels, mid_channels, 3, stride=2, use_lab=use_lab)
        self.stem2a = HGConvBN(mid_channels, mid_channels // 2, 2, use_lab=use_lab)
        self.stem2b = HGConvBN(mid_channels // 2, mid_channels, 2, use_lab=use_lab)
        self.stem3 = HGConvBN(mid_channels * 2, mid_channels, 3, stride=2, use_lab=use_lab)
        self.stem4 = HGConvBN(mid_channels, out_channels, 1, use_lab=use_lab)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def forward(
        self, x: Float[torch.Tensor, "batch_size in_channels height width"]
    ) -> Float[torch.Tensor, "batch_size out_channels quarter_height quarter_width"]:
        """Reduce the image to the stride 4 feature map."""
        x = self.stem1(x)
        # The 2x2 kernels are unpadded: one row and one column keep the resolution.
        x = F.pad(x, (0, 1, 0, 1))
        branch = self.stem2a(x)
        branch = F.pad(branch, (0, 1, 0, 1))
        branch = self.stem2b(branch)
        x = torch.cat([self.pool(x), branch], dim=1)
        return self.stem4(self.stem3(x))


class HGBlock(nn.Module):
    """A chain of layers whose input and every intermediate output are concatenated and fused."""

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        num_layers: int,
        kernel_size: int,
        residual: bool,
        light_block: bool,
        use_lab: bool = True,
    ) -> None:
        """Initialize the block.

        Args:
            in_channels: Input channel count.
            mid_channels: Channels of every layer of the chain.
            out_channels: Output channel count.
            num_layers: Number of layers of the chain.
            kernel_size: Kernel size of the layers.
            residual: Whether the input is added to the output.
            light_block: Whether the layers are the pointwise plus depthwise ones.
            use_lab: Whether a learnable affine block follows every ReLU.
        """
        super().__init__()
        self.residual = residual
        self.layers = nn.ModuleList()
        for index in range(num_layers):
            layer_in = in_channels if index == 0 else mid_channels
            if light_block:
                self.layers.append(HGLightConvBN(layer_in, mid_channels, kernel_size, use_lab))
            else:
                self.layers.append(HGConvBN(layer_in, mid_channels, kernel_size, use_lab=use_lab))
        total_channels = in_channels + num_layers * mid_channels
        self.aggregation = nn.Sequential(
            HGConvBN(total_channels, out_channels // 2, 1, use_lab=use_lab),
            HGConvBN(out_channels // 2, out_channels, 1, use_lab=use_lab),
        )

    def forward(
        self, x: Float[torch.Tensor, "batch_size in_channels height width"]
    ) -> Float[torch.Tensor, "batch_size out_channels height width"]:
        """Run the chain and fuse its concatenated outputs."""
        outputs = [x]
        y = x
        for layer in self.layers:
            y = layer(y)
            outputs.append(y)
        y = self.aggregation(torch.cat(outputs, dim=1))
        return y + x if self.residual else y


class HGStage(nn.Module):
    """An optional depthwise stride 2 convolution followed by the blocks of the stage."""

    def __init__(self, in_channels: int, spec: _StageSpec, use_lab: bool = True) -> None:
        """Initialize the stage.

        Args:
            in_channels: Input channel count.
            spec: Layout of the stage.
            use_lab: Whether a learnable affine block follows every ReLU.
        """
        super().__init__()
        self.downsample: nn.Module = (
            HGConvBN(
                in_channels,
                in_channels,
                3,
                stride=2,
                groups=in_channels,
                use_act=False,
                use_lab=False,
            )
            if spec.downsample
            else nn.Identity()
        )
        self.blocks = nn.Sequential(
            *(
                HGBlock(
                    in_channels if index == 0 else spec.out_channels,
                    spec.mid_channels,
                    spec.out_channels,
                    spec.num_layers,
                    spec.kernel_size,
                    residual=index > 0,
                    light_block=spec.light_block,
                    use_lab=use_lab,
                )
                for index in range(spec.num_blocks)
            )
        )

    def forward(
        self, x: Float[torch.Tensor, "batch_size in_channels height width"]
    ) -> Float[torch.Tensor, "batch_size out_channels out_height out_width"]:
        """Downsample and run the blocks."""
        return self.blocks(self.downsample(x))


class HGNetV2(nn.Module):
    """HGNetV2 with a stride 4 stem and four stages at strides 4, 8, 16 and 32.

    The stages of B0 and B1 output 64, 256, 512 and 1024 channels. ``projection_channels``
    adds a 1x1 convolution block behind every returned stage, which lets a neck built for
    narrower feature maps (128, 256 and 512 for YOLOX-S) take this backbone unchanged. The
    projections are new layers a pretrained state dict does not hold.
    """

    def __init__(
        self,
        variant: str = "b0",
        in_channels: int = 3,
        out_indices: Sequence[int] = (1, 2, 3),
        projection_channels: Sequence[int] | None = None,
        projection_activation: str = "silu",
        weights: str | Path | None = None,
    ) -> None:
        """Initialize the backbone.

        Args:
            variant: ``b0`` or ``b1``.
            in_channels: Channels of the input image.
            out_indices: Stages whose feature maps are returned, 0 (stride 4) to 3 (stride 32).
            projection_channels: Output channels of a 1x1 convolution block per returned
                stage, None to return the stage outputs as they are.
            projection_activation: Activation name of the projection blocks.
            weights: Pretrained weights of the stem and the stages: the name of a weight set
                of the variant, such as ``ssld_stage2_ft_in1k``, which is downloaded into the
                torch hub cache, or the path of a timm state dict file. ``None`` keeps the
                random initialization.

        Raises:
            ValueError: If the variant is unknown, ``out_indices`` is empty, not increasing
                or outside the stages, or ``projection_channels`` does not name one width per
                returned stage.
        """
        super().__init__()
        if variant not in _VARIANTS:
            raise ValueError(f"Unknown HGNetV2 variant {variant!r}, expected {sorted(_VARIANTS)}.")
        indices = tuple(int(index) for index in out_indices)
        if not indices or any(not 0 <= index <= 3 for index in indices):
            raise ValueError(f"out_indices must select stages in [0, 3], got {indices}.")
        if any(later <= earlier for earlier, later in zip(indices, indices[1:])):
            raise ValueError(f"out_indices must be strictly increasing, got {indices}.")
        self.variant = variant
        self.out_indices = indices
        spec = _VARIANTS[variant]

        self.stem = HGStem(in_channels, spec.stem_channels[0], spec.stem_channels[1])
        stages = []
        stage_in = spec.stem_channels[1]
        for stage_spec in spec.stages:
            stages.append(HGStage(stage_in, stage_spec))
            stage_in = stage_spec.out_channels
        self.stages = nn.Sequential(*stages)
        stage_channels = tuple(spec.stages[index].out_channels for index in indices)

        if projection_channels is None:
            self.projections = None
            self.out_channels = stage_channels
        else:
            widths = tuple(int(channels) for channels in projection_channels)
            if len(widths) != len(indices):
                raise ValueError(
                    f"projection_channels names {len(widths)} widths for {len(indices)} "
                    "returned stages."
                )
            self.projections = nn.ModuleList(
                ConvBNAct(channels, width, 1, 1, activation=projection_activation)
                for channels, width in zip(stage_channels, widths)
            )
            self.out_channels = widths

        if weights is not None:
            self.load_pretrained_weights(weights)

    def load_pretrained_weights(self, weights: str | Path) -> None:
        """Load a timm state dict into the stem and the stages.

        The classification head of the state dict is dropped. The stem convolution is skipped
        when the backbone reads another number of channels than the state dict. The
        projection blocks keep their initialization.

        Args:
            weights: Name of a weight set of the variant or path of a timm state dict file.

        Raises:
            ValueError: If the name is no weight set of the variant, or the state dict does
                not hold exactly the stem and stage parameters besides the head.
        """
        path = Path(weights)
        if path.is_file():
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            source = str(path)
        else:
            urls = _VARIANTS[self.variant].weight_urls
            if str(weights) not in urls:
                raise ValueError(
                    f"weights={weights!r} is neither a state dict file nor one of the weight "
                    f"sets {sorted(urls)} of HGNetV2-{self.variant.upper()}."
                )
            source = f"hgnetv2_{self.variant}.{weights}"
            state_dict = torch.hub.load_state_dict_from_url(
                urls[str(weights)],
                map_location="cpu",
                progress=False,
                file_name=f"{source}.bin",
                weights_only=True,
            )
        if not isinstance(state_dict, dict):
            raise ValueError(f"{source} does not hold a state dict.")
        state_dict = {
            key: value for key, value in state_dict.items() if not key.startswith("head.")
        }
        stem_key = "stem.stem1.conv.weight"
        checkpoint_stem = state_dict.get(stem_key)
        own_stem = self.stem.stem1.conv.weight
        if checkpoint_stem is not None and checkpoint_stem.shape != own_stem.shape:
            logger.warning(
                "Skipping the stem convolution of %s: it reads %d channels while this backbone "
                "reads %d.",
                source,
                checkpoint_stem.shape[1],
                own_stem.shape[1],
            )
            # Keep the backbone's own stem, so the check below still covers every other key.
            state_dict[stem_key] = own_stem.detach().clone()
        try:
            result = self.load_state_dict(state_dict, strict=False)
        except RuntimeError as error:
            raise ValueError(
                f"{source} does not hold the parameters of HGNetV2-{self.variant.upper()}: {error}"
            ) from error
        missing = [key for key in result.missing_keys if not key.startswith("projections.")]
        if missing or result.unexpected_keys:
            raise ValueError(
                f"{source} does not hold the parameters of HGNetV2-{self.variant.upper()}: "
                f"missing {missing}, unexpected {list(result.unexpected_keys)}."
            )
        logger.info("Loaded %d tensors of %s.", len(state_dict), source)

    def forward(
        self, x: Float[torch.Tensor, "batch_size in_channels height width"]
    ) -> tuple[Float[torch.Tensor, "batch_size channels level_height level_width"], ...]:
        """Extract the feature maps of the stages selected by ``out_indices``.

        Args:
            x: Images, their height and width divisible by 32.

        Returns:
            One feature map per selected stage, from the shallowest to the deepest.
        """
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        selected = [features[index] for index in self.out_indices]
        if self.projections is not None:
            selected = [
                projection(feature)
                for projection, feature in zip(self.projections, selected, strict=True)
            ]
        return tuple(selected)
