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

"""CSPNeXt backbone of RTMDet.

Ported from MMDetection (``mmdet/models/backbones/cspnext.py`` and
``mmdet/models/layers/csp_layer.py`` of https://github.com/open-mmlab/mmdetection,
Apache-2.0). The submodule names follow the reference, so the backbone part of its
checkpoints loads without a key rename. The backbone follows the multi-scale tuple interface
of the other image backbones: it returns the feature maps of the selected stages, by default
those at strides 8, 16 and 32.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging
from pathlib import Path
import pickle
import types
from typing import Any

from jaxtyping import Float
import torch
import torch.nn as nn

from autoware_ml.models.common.layers.csp import build_activation

logger = logging.getLogger(__name__)

_DOWNLOAD_URL = "https://download.openmmlab.com/mmdetection/v3.0/rtmdet"

# Published checkpoints of MMDetection (Apache-2.0) holding a CSPNeXt-S backbone: the name,
# the file and the leading characters of its SHA-256. The ImageNet classifier reads RGB, the
# COCO detector BGR, both normalized by the ImageNet statistics.
_WEIGHT_SETS: dict[str, tuple[str, str]] = {
    "cspnext_s_imagenet_600e": (
        f"{_DOWNLOAD_URL}/cspnext_rsb_pretrain/cspnext-s_imagenet_600e-ea671761.pth",
        "ea671761",
    ),
    "rtmdet_s_coco_300e": (
        f"{_DOWNLOAD_URL}/rtmdet_s_8xb32-300e_coco/"
        "rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth",
        "387a891e",
    ),
}


class _Skipped:
    """Stands in for every object of a checkpoint that is not a tensor container."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Accept and drop the constructor arguments of the replaced object."""

    def __call__(self, *args: Any, **kwargs: Any) -> _Skipped:
        """Stand in for a replaced reconstruction function."""
        return _Skipped()

    def __setstate__(self, state: Any) -> None:
        """Drop the state of the replaced object."""

    def append(self, item: Any) -> None:
        """Drop an item of a replaced list-like object."""

    def extend(self, items: Any) -> None:
        """Drop the items of a replaced list-like object."""

    def __setitem__(self, key: Any, value: Any) -> None:
        """Drop an entry of a replaced dict-like object."""


class _TensorOnlyUnpickler(pickle.Unpickler):
    """Unpickler that builds tensors and ordered dicts and replaces every other object.

    A training checkpoint of MMDetection holds the logging state of the training framework
    next to the weights, which a weights-only load refuses. No code of the file runs here:
    only the tensor reconstruction of torch is resolved, everything else becomes a
    placeholder.
    """

    def find_class(self, module: str, name: str) -> Any:
        """Resolve the tensor classes of torch and replace everything else."""
        if (module, name) == ("collections", "OrderedDict") or module.split(".")[0] == "torch":
            return super().find_class(module, name)
        return _Skipped


def _load_tensor_only(file: Any, **kwargs: Any) -> Any:
    """Unpickle with :class:`_TensorOnlyUnpickler`, the ``load`` of the pickle module below."""
    return _TensorOnlyUnpickler(file, **kwargs).load()


_TENSOR_ONLY_PICKLE = types.SimpleNamespace(
    __name__="tensor_only_pickle", Unpickler=_TensorOnlyUnpickler, load=_load_tensor_only
)


def load_backbone_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Read the backbone tensors of an MMDetection or MMPretrain checkpoint file.

    Args:
        path: Checkpoint file, a state dict or a checkpoint holding one under ``state_dict``.

    Returns:
        The tensors below the ``backbone.`` prefix without the prefix, or every tensor when
        no key carries the prefix.

    Raises:
        ValueError: If the file holds no state dict.
    """
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=False, pickle_module=_TENSOR_ONLY_PICKLE
        )
    state_dict = (
        checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, Mapping) else None
    )
    if not isinstance(state_dict, Mapping) or not all(
        isinstance(value, torch.Tensor) for value in state_dict.values()
    ):
        raise ValueError(f"{path} does not hold a state dict.")
    prefix = "backbone."
    if any(key.startswith(prefix) for key in state_dict):
        return {
            key.removeprefix(prefix): value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
    return dict(state_dict)


class ConvModule(nn.Module):
    """Convolution, batch normalization and activation, named as in the reference."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        activation: str = "silu",
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
    ) -> None:
        """Initialize the block.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            kernel_size: Kernel size, padded so a stride of 1 keeps the resolution.
            stride: Convolution stride.
            groups: Convolution groups.
            activation: Activation name.
            bn_eps: Epsilon of the batch normalization.
            bn_momentum: Momentum of the batch normalization.
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
        self.bn = nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_momentum)
        self.activate = build_activation(activation)

    def forward(
        self, x: Float[torch.Tensor, "batch_size in_channels height width"]
    ) -> Float[torch.Tensor, "batch_size out_channels out_height out_width"]:
        """Apply convolution, normalization and activation."""
        return self.activate(self.bn(self.conv(x)))


class DepthwiseSeparableConvModule(nn.Module):
    """A depthwise and a pointwise convolution block."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, **conv_kwargs: Any
    ) -> None:
        """Initialize the block.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            kernel_size: Kernel size of the depthwise convolution.
            **conv_kwargs: Activation and normalization arguments of :class:`ConvModule`.
        """
        super().__init__()
        self.depthwise_conv = ConvModule(
            in_channels, in_channels, kernel_size, groups=in_channels, **conv_kwargs
        )
        self.pointwise_conv = ConvModule(in_channels, out_channels, 1, **conv_kwargs)

    def forward(
        self, x: Float[torch.Tensor, "batch_size in_channels height width"]
    ) -> Float[torch.Tensor, "batch_size out_channels height width"]:
        """Apply the depthwise and the pointwise convolution."""
        return self.pointwise_conv(self.depthwise_conv(x))


class CSPNeXtBlock(nn.Module):
    """A 3x3 convolution and a 5x5 depthwise separable one with an optional residual."""

    def __init__(self, channels: int, add_identity: bool, **conv_kwargs: Any) -> None:
        """Initialize the block.

        Args:
            channels: Input and output channel count.
            add_identity: Whether the input is added to the output.
            **conv_kwargs: Activation and normalization arguments of :class:`ConvModule`.
        """
        super().__init__()
        self.conv1 = ConvModule(channels, channels, 3, **conv_kwargs)
        self.conv2 = DepthwiseSeparableConvModule(channels, channels, 5, **conv_kwargs)
        self.add_identity = add_identity

    def forward(
        self, x: Float[torch.Tensor, "batch_size channels height width"]
    ) -> Float[torch.Tensor, "batch_size channels height width"]:
        """Apply the two convolutions and the residual connection."""
        y = self.conv2(self.conv1(x))
        return y + x if self.add_identity else y


class ChannelAttention(nn.Module):
    """Scale every channel by a gate computed from the channel means."""

    def __init__(self, channels: int) -> None:
        """Initialize the gate.

        Args:
            channels: Channel count of the feature map.
        """
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(
        self, x: Float[torch.Tensor, "batch_size channels height width"]
    ) -> Float[torch.Tensor, "batch_size channels height width"]:
        """Gate the channels."""
        return x * self.act(self.fc(self.global_avgpool(x)))


class CSPNeXtLayer(nn.Module):
    """Cross stage partial layer of CSPNeXt blocks with a channel attention on the fusion."""

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        add_identity: bool,
        expand_ratio: float = 0.5,
        channel_attention: bool = True,
        **conv_kwargs: Any,
    ) -> None:
        """Initialize the layer.

        Args:
            channels: Input and output channel count.
            num_blocks: Number of blocks.
            add_identity: Whether the blocks are residual.
            expand_ratio: Channels of the two branches as a share of the output channels.
            channel_attention: Whether the concatenated branches pass a channel attention.
            **conv_kwargs: Activation and normalization arguments of :class:`ConvModule`.
        """
        super().__init__()
        mid_channels = int(channels * expand_ratio)
        self.main_conv = ConvModule(channels, mid_channels, 1, **conv_kwargs)
        self.short_conv = ConvModule(channels, mid_channels, 1, **conv_kwargs)
        self.final_conv = ConvModule(2 * mid_channels, channels, 1, **conv_kwargs)
        self.blocks = nn.Sequential(
            *(CSPNeXtBlock(mid_channels, add_identity, **conv_kwargs) for _ in range(num_blocks))
        )
        if channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)
        self.channel_attention = channel_attention

    def forward(
        self, x: Float[torch.Tensor, "batch_size channels height width"]
    ) -> Float[torch.Tensor, "batch_size channels height width"]:
        """Fuse the block branch with the bypass branch."""
        fused = torch.cat((self.blocks(self.main_conv(x)), self.short_conv(x)), dim=1)
        if self.channel_attention:
            fused = self.attention(fused)
        return self.final_conv(fused)


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling: max pools of several kernel sizes concatenated."""

    def __init__(
        self,
        channels: int,
        kernel_sizes: Sequence[int] = (5, 9, 13),
        **conv_kwargs: Any,
    ) -> None:
        """Initialize the pooling block.

        Args:
            channels: Input and output channel count.
            kernel_sizes: Kernel sizes of the max pools.
            **conv_kwargs: Activation and normalization arguments of :class:`ConvModule`.
        """
        super().__init__()
        mid_channels = channels // 2
        self.conv1 = ConvModule(channels, mid_channels, 1, **conv_kwargs)
        self.poolings = nn.ModuleList(
            nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            for kernel_size in kernel_sizes
        )
        self.conv2 = ConvModule(mid_channels * (len(kernel_sizes) + 1), channels, 1, **conv_kwargs)

    def forward(
        self, x: Float[torch.Tensor, "batch_size channels height width"]
    ) -> Float[torch.Tensor, "batch_size channels height width"]:
        """Pool the reduced features at every kernel size and fuse them."""
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [pool(x) for pool in self.poolings], dim=1))


class CSPNeXt(nn.Module):
    """CSPNeXt with a stride 2 stem of three convolutions and four downsampling stages.

    ``deepen_factor`` and ``widen_factor`` scale the block counts and the channel widths of
    the base network: RTMDet-tiny is 0.167 / 0.375, S 0.33 / 0.5, M 0.67 / 0.75 and L
    1.0 / 1.0.
    """

    def __init__(
        self,
        deepen_factor: float = 0.33,
        widen_factor: float = 0.5,
        in_channels: int = 3,
        out_indices: Sequence[int] = (1, 2, 3),
        expand_ratio: float = 0.5,
        channel_attention: bool = True,
        activation: str = "silu",
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
        weights: str | Path | None = None,
    ) -> None:
        """Initialize the backbone.

        Args:
            deepen_factor: Multiplier of the number of blocks of every stage.
            widen_factor: Multiplier of the channel width of every stage.
            in_channels: Channels of the input image.
            out_indices: Stages whose feature maps are returned, 0 (stride 4) to 3 (stride 32).
            expand_ratio: Channels of the branches of a stage as a share of its width.
            channel_attention: Whether every stage ends with a channel attention.
            activation: Activation name of every block.
            bn_eps: Epsilon of every batch normalization, that of the published weights.
            bn_momentum: Momentum of every batch normalization.
            weights: Pretrained weights: the name of a published checkpoint,
                ``cspnext_s_imagenet_600e`` or ``rtmdet_s_coco_300e``, which is downloaded
                into the torch hub cache, or the path of a checkpoint file. Only the backbone
                of a detector checkpoint is read. ``None`` keeps the random initialization.

        Raises:
            ValueError: If ``out_indices`` is empty, not increasing or outside the stages.
        """
        super().__init__()
        indices = tuple(int(index) for index in out_indices)
        if not indices or any(not 0 <= index <= 3 for index in indices):
            raise ValueError(f"out_indices must select stages in [0, 3], got {indices}.")
        if any(later <= earlier for earlier, later in zip(indices, indices[1:])):
            raise ValueError(f"out_indices must be strictly increasing, got {indices}.")
        self.out_indices = indices
        conv_kwargs = {"activation": activation, "bn_eps": bn_eps, "bn_momentum": bn_momentum}

        base_channels = int(64 * widen_factor)
        channels = [base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
        num_blocks = [max(round(blocks * deepen_factor), 1) for blocks in (3, 6, 6, 3)]
        self.out_channels = tuple(channels[index] for index in indices)

        self.stem = nn.Sequential(
            ConvModule(in_channels, base_channels // 2, 3, stride=2, **conv_kwargs),
            ConvModule(base_channels // 2, base_channels // 2, 3, **conv_kwargs),
            ConvModule(base_channels // 2, base_channels, 3, **conv_kwargs),
        )
        stage_in = base_channels
        for index, (stage_out, blocks) in enumerate(zip(channels, num_blocks)):
            last = index == 3
            layers: list[nn.Module] = [ConvModule(stage_in, stage_out, 3, stride=2, **conv_kwargs)]
            if last:
                layers.append(SPPBottleneck(stage_out, **conv_kwargs))
            layers.append(
                CSPNeXtLayer(
                    stage_out,
                    blocks,
                    add_identity=not last,
                    expand_ratio=expand_ratio,
                    channel_attention=channel_attention,
                    **conv_kwargs,
                )
            )
            self.add_module(f"stage{index + 1}", nn.Sequential(*layers))
            stage_in = stage_out

        if weights is not None:
            self.load_pretrained_weights(weights)

    def load_pretrained_weights(self, weights: str | Path) -> None:
        """Load the backbone of a published or a local checkpoint.

        The stem convolution is skipped when the backbone reads another number of channels
        than the checkpoint.

        Args:
            weights: Name of a published checkpoint or path of a checkpoint file.

        Raises:
            ValueError: If the name is no published checkpoint, or the checkpoint does not
                hold exactly the parameters of this backbone.
        """
        path = Path(weights)
        if path.is_file():
            source = str(path)
        else:
            if str(weights) not in _WEIGHT_SETS:
                raise ValueError(
                    f"weights={weights!r} is neither a checkpoint file nor one of the "
                    f"published checkpoints {sorted(_WEIGHT_SETS)}."
                )
            url, hash_prefix = _WEIGHT_SETS[str(weights)]
            path = Path(torch.hub.get_dir()) / "checkpoints" / url.rsplit("/", 1)[1]
            if not path.is_file():
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.hub.download_url_to_file(
                    url, str(path), hash_prefix=hash_prefix, progress=False
                )
            source = str(weights)
        state_dict = load_backbone_state_dict(path)
        stem_key = "stem.0.conv.weight"
        checkpoint_stem = state_dict.get(stem_key)
        own_stem = self.stem[0].conv.weight
        if checkpoint_stem is not None and checkpoint_stem.shape != own_stem.shape:
            logger.warning(
                "Skipping the stem convolution of %s: it reads %d channels while this backbone "
                "reads %d.",
                source,
                checkpoint_stem.shape[1],
                own_stem.shape[1],
            )
            # Keep the backbone's own stem, so a strict load still checks every other key.
            state_dict[stem_key] = own_stem.detach().clone()
        try:
            self.load_state_dict(state_dict, strict=True)
        except RuntimeError as error:
            raise ValueError(
                f"{source} does not hold the parameters of this backbone: {error}"
            ) from error
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
        stages = []
        for stage in (self.stage1, self.stage2, self.stage3, self.stage4):
            x = stage(x)
            stages.append(x)
        return tuple(stages[index] for index in self.out_indices)
