"""
PyTorch port of SST-EmotionNet backbone (Jia et al., ACM MM 2020).

Tensor convention throughout: (N, C, D, H, W).
Keras kernel (kH, kW, kD) channels_last maps to PyTorch (kD, kH, kW).

Bug fix vs original Keras code (model.py:151): the original applied the initial
Conv3D twice directly to img_input (not chained). Here both convs are chained.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """BN → ReLU → [bottleneck 1×1×1] → Conv3d(1,3,3) → Conv3d(3,1,1) [→ Dropout]."""

    def __init__(self, in_channels: int, growth_rate: int,
                 bottleneck: bool = False, dropout_rate: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = [
            nn.BatchNorm3d(in_channels, eps=1.1e-5),
            nn.ReLU(inplace=True),
        ]
        ch = in_channels
        if bottleneck:
            inter = growth_rate * 4
            layers += [
                nn.Conv3d(ch, inter, 1, bias=False),
                nn.BatchNorm3d(inter, eps=1.1e-5),
                nn.ReLU(inplace=True),
            ]
            ch = inter

        # Decomposed 3D conv: Keras (3,3,1)→(1,1,3) ≡ PyTorch (1,3,3)→(3,1,1)
        layers += [
            nn.Conv3d(ch, growth_rate, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.Conv3d(growth_rate, growth_rate, (3, 1, 1), padding=(1, 0, 0), bias=False),
        ]
        if dropout_rate > 0:
            layers.append(nn.Dropout3d(dropout_rate))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DenseBlock(nn.Module):
    """Each layer receives the concatenation of all prior feature maps."""

    def __init__(self, in_channels: int, nb_layers: int, growth_rate: int,
                 bottleneck: bool = False, dropout_rate: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        ch = in_channels
        for _ in range(nb_layers):
            self.layers.append(ConvBlock(ch, growth_rate, bottleneck, dropout_rate))
            ch += growth_rate
        self.out_channels = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        return x


class TransitionBlock(nn.Module):
    """BN → ReLU → Conv3d(1,1,1) compression → AvgPool3d(2,2,2)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm3d(in_channels, eps=1.1e-5),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool3d(2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionBlock(nn.Module):
    """
    Spatial-temporal self-attention gate (sigmoid-gated linear on channel mean).

    spatial_size  = H * W at this stage.
    temporal_size = D     at this stage.
    """

    def __init__(self, spatial_size: int, temporal_size: int,
                 spatial: bool = True, temporal: bool = True):
        super().__init__()
        self.do_spatial = spatial and spatial_size > 0
        self.do_temporal = temporal and temporal_size > 0
        self.spatial_fc = nn.Linear(spatial_size, spatial_size) if self.do_spatial else None
        self.temporal_fc = nn.Linear(temporal_size, temporal_size) if self.do_temporal else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, D, H, W = x.shape
        out = x
        x_mean = x.mean(dim=1, keepdim=True)          # (N, 1, D, H, W)

        if self.do_spatial and self.spatial_fc is not None:
            s = F.adaptive_avg_pool3d(x_mean, (1, H, W)).view(N, H * W)
            s = torch.sigmoid(self.spatial_fc(s)).view(N, 1, 1, H, W)
            out = out * s

        if self.do_temporal and self.temporal_fc is not None:
            t = F.adaptive_avg_pool3d(x_mean, (D, 1, 1)).view(N, D)
            t = torch.sigmoid(self.temporal_fc(t)).view(N, 1, D, 1, 1)
            out = out * t

        return out


# ---------------------------------------------------------------------------
# Single stream (spectral or temporal)
# ---------------------------------------------------------------------------

def _attn_sizes_after_transitions(
    D: int, H: int, W: int, nb_dense_block: int
) -> List[Tuple[int, int]]:
    """Compute (spatial_size, temporal_size) after each of the nb_dense_block-1 transitions."""
    sizes = []
    for _ in range(nb_dense_block - 1):
        D = D // 2
        H = H // 2
        W = W // 2
        sizes.append((H * W, D))
    return sizes


class SSTStream(nn.Module):
    """
    One 3D DenseNet stream with spatial-temporal attention after each transition.

    subsample_initial_block=True  → temporal stream (large stride + MaxPool at start).
    subsample_initial_block=False → spectral stream (plain initial conv).
    """

    def __init__(
        self,
        in_channels: int = 1,
        depth: int = 16,
        nb_dense_block: int = 3,
        growth_rate: int = 12,
        reduction: float = 0.5,
        bottleneck: bool = True,
        dropout_rate: float = 0.0,
        subsample_initial_block: bool = False,
        use_attention: bool = True,
        spatial_attention: bool = True,
        temporal_attention: bool = True,
        input_D: int = 5,
        input_H: int = 32,
        input_W: int = 32,
    ):
        super().__init__()
        count = (depth - 4) // 3
        if bottleneck:
            count = count // 2
        nb_layers = [count] * nb_dense_block

        nb_filter = 2 * growth_rate
        compression = 1.0 - reduction

        # ---- Initial convolution ----
        if subsample_initial_block:
            # Keras kernel (5,5,3) stride (2,2,1) → PyTorch (3,5,5) stride (1,2,2)
            self.initial = nn.Sequential(
                nn.Conv3d(in_channels, nb_filter, (3, 5, 5),
                          stride=(1, 2, 2), padding=(1, 2, 2), bias=False),
                nn.BatchNorm3d(nb_filter, eps=1.1e-5),
                nn.ReLU(inplace=True),
                # ceil_mode=True matches Keras padding='same' on odd dimensions
                nn.MaxPool3d(2, stride=2, ceil_mode=True),
            )
            # Track feature-map shape after initial block.
            # Conv3d stride (1,2,2): D unchanged, H and W halved.
            # MaxPool3d(2, ceil_mode=True): all dims halved (ceil).
            after_conv_D = input_D                        # stride-1 on D
            after_conv_H = math.ceil(input_H / 2)        # stride-2 on H
            after_conv_W = math.ceil(input_W / 2)        # stride-2 on W
            init_D = math.ceil(after_conv_D / 2)
            init_H = math.ceil(after_conv_H / 2)
            init_W = math.ceil(after_conv_W / 2)
        else:
            # Bug fix: chain the two initial convs (Keras code applied both to img_input)
            self.initial = nn.Sequential(
                nn.Conv3d(in_channels, nb_filter, (1, 3, 3),
                          stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
                nn.Conv3d(nb_filter, nb_filter, (1, 3, 3),
                          stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            )
            init_D, init_H, init_W = input_D, input_H, input_W

        # ---- Dense blocks + transitions + attention ----
        attn_sizes = _attn_sizes_after_transitions(init_D, init_H, init_W, nb_dense_block)

        self.body = nn.ModuleList()
        ch = nb_filter
        for block_idx in range(nb_dense_block):
            dense = DenseBlock(ch, nb_layers[block_idx], growth_rate, bottleneck, dropout_rate)
            self.body.append(dense)
            ch = dense.out_channels

            if block_idx < nb_dense_block - 1:
                out_ch = int(ch * compression)
                self.body.append(TransitionBlock(ch, out_ch))
                ch = out_ch

                if use_attention:
                    sp, tmp = attn_sizes[block_idx]
                    self.body.append(
                        AttentionBlock(sp, tmp, spatial_attention, temporal_attention)
                    )

        self.final_norm = nn.Sequential(
            nn.BatchNorm3d(ch, eps=1.1e-5),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.out_channels = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        for block in self.body:
            x = block(x)
        x = self.final_norm(x)
        return self.gap(x).flatten(1)


# ---------------------------------------------------------------------------
# Full dual-stream model
# ---------------------------------------------------------------------------

class SSTEmotionNet(nn.Module):
    """
    Dual-stream SST-EmotionNet.

    Input A (spectral): (N, 1, spec_length, input_width, input_width)
    Input B (temporal): (N, 1, tem_length,  input_width, input_width)

    The model exposes `features` and `head` as named sub-modules to support
    ANIL (inner-loop updates head only).
    """

    def __init__(
        self,
        input_width: int = 32,
        spec_length: int = 5,
        tem_length: int = 25,
        depth_spec: int = 16,
        depth_tem: int = 22,
        gr_spec: int = 12,
        gr_tem: int = 24,
        nb_dense_block: int = 3,
        nb_class: int = 3,
        use_attention: bool = True,
        spatial_attention: bool = True,
        temporal_attention: bool = True,
    ):
        super().__init__()

        self.spec_stream = SSTStream(
            in_channels=1,
            depth=depth_spec,
            nb_dense_block=nb_dense_block,
            growth_rate=gr_spec,
            reduction=0.5,
            bottleneck=True,
            subsample_initial_block=False,
            use_attention=use_attention,
            spatial_attention=spatial_attention,
            temporal_attention=temporal_attention,
            input_D=spec_length,
            input_H=input_width,
            input_W=input_width,
        )

        self.temp_stream = SSTStream(
            in_channels=1,
            depth=depth_tem,
            nb_dense_block=nb_dense_block,
            growth_rate=gr_tem,
            reduction=0.5,
            bottleneck=True,
            subsample_initial_block=True,
            use_attention=use_attention,
            spatial_attention=spatial_attention,
            temporal_attention=temporal_attention,
            input_D=tem_length,
            input_H=input_width,
            input_W=input_width,
        )

        feat_dim = self.spec_stream.out_channels + self.temp_stream.out_channels
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 50),
            nn.Dropout(0.5),
            nn.Linear(50, nb_class),
        )

    # ------------------------------------------------------------------
    # Forward helpers — kept separate for ANIL support
    # ------------------------------------------------------------------

    def get_features(self, spec: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.spec_stream(spec), self.temp_stream(temp)], dim=1)

    def forward(self, spec: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
        return self.head(self.get_features(spec, temp))


# ---------------------------------------------------------------------------
# Factory matching the config .ini parameters
# ---------------------------------------------------------------------------

def build_model(cfg: dict, device: torch.device) -> SSTEmotionNet:
    """Build SSTEmotionNet from a flat config dict and move to device."""
    model = SSTEmotionNet(
        input_width=int(cfg["input_width"]),
        spec_length=int(cfg["specInput_length"]),
        tem_length=int(cfg["temInput_length"]),
        depth_spec=int(cfg["depth_spec"]),
        depth_tem=int(cfg["depth_tem"]),
        gr_spec=int(cfg["gr_spec"]),
        gr_tem=int(cfg["gr_tem"]),
        nb_dense_block=int(cfg["nb_dense_block"]),
        nb_class=int(cfg["nb_class"]),
    )
    return model.to(device)
