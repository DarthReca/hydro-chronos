import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from timm.layers.create_act import create_act_layer

from .convlstm import ConvLSTM


class ACTU(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size,
        padding,
        stride,
        backbone: str,
        bias=True,
        batch_first=True,
        bidirectional=False,
        original_resolution=(256, 256),
        act_layer: str = "sigmoid",
        n_classes: int = 1,
        **kwargs,
    ):
        super(ACTU, self).__init__()
        self.n_classes = n_classes
        self.backbone = backbone

        self.encoder: nn.Module = timm.create_model(
            backbone, features_only=True, in_chans=in_channels
        )

        with torch.no_grad():
            embs = self.encoder.forward(
                torch.randn(1, in_channels, *original_resolution)
            )
            embs_shape = [e.shape for e in embs]

        # The ConvLSTM expects inputs of shape (B, T, feature_dim, H_enc, W_enc)
        # We assume the provided ConvLSTM code is available.
        self.convlstm = nn.ModuleList(
            ConvLSTM(
                in_channels=shape[1],
                hidden_channels=shape[1],
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=bias,
                batch_first=batch_first,
                bidirectional=bidirectional,
            )
            for shape in embs_shape
        )
        # If bidirectional, the hidden representation is concatenated from both directions.
        n_upsamples = int(np.log2(original_resolution[0] / embs_shape[-1][-2]))
        skip_channels_list = [shape[1] for shape in embs_shape[-(n_upsamples + 1) : -1]]
        skip_channels_list = skip_channels_list[::-1]  # Reverse the list.
        encoder_channels = [e[1] for e in embs_shape]

        self.decoder = UnetDecoder(
            encoder_channels=[1, *encoder_channels],
            decoder_channels=encoder_channels[::-1],
            n_blocks=len(encoder_channels),
        )
        self.seg_head = nn.Sequential(
            SegmentationHead(
                in_channels=encoder_channels[0],
                out_channels=n_classes,
            ),
            create_act_layer(act_layer, inplace=True),
        )
        self.encoder_channels = encoder_channels
        self.embs_shape = embs_shape

    def forward(self, x: torch.Tensor, **kwargs):
        size = x.size()[-2:]
        # Process each time step through the encoder.
        x = self._encode_images(x)
        # Pass the encoded sequence through the ConvLSTM.
        x = self._encode_timeseries(x)
        return self._decode(x, size=size)

    def _encode_images(self, x: torch.Tensor) -> list[torch.Tensor]:
        B = x.size(0)
        encoded_frames = self.encoder(rearrange(x, "b t c h w -> (b t) c h w"))
        return [
            rearrange(frames, "(b t) c h w -> b t c h w", b=B)
            for frames in encoded_frames
        ]

    def _encode_timeseries(self, timeseries: torch.Tensor) -> list[torch.Tensor]:
        outs = []
        for convlstm, encoded in reversed(list(zip(self.convlstm, timeseries))):
            lstm_out, (_, _) = convlstm(encoded)
            outs.append(lstm_out[:, -1, :, :, :])
        return outs

    def _decode(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        trend_map = self.decoder(*[None] + x[::-1])
        trend_map = self.seg_head(trend_map)
        trend_map = F.interpolate(
            trend_map, size=size, mode="bilinear", align_corners=False
        )
        return trend_map
