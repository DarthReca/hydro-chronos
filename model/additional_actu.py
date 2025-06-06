import numpy as np
import torch
from einops import rearrange, repeat
from torch import nn

from .actu import ACTU


class ClimateACTU(ACTU):
    def __init__(
        self,
        in_channels,
        kernel_size,
        padding,
        stride,
        backbone,
        bias=True,
        batch_first=True,
        bidirectional=False,
        original_resolution=(256, 256),
        act_layer="sigmoid",
        seq_len=5,
        n_classes=1,
    ):
        super().__init__(
            in_channels,
            kernel_size,
            padding,
            stride,
            backbone,
            bias,
            batch_first,
            bidirectional,
            original_resolution,
            act_layer,
            n_classes,
        )
        self.climate_branch = ClimateBranchLSTM(
            [e[1:] for e in self.embs_shape],
            lstm_hidden_dim=128,
            climate_seq_len=seq_len,
            climate_input_dim=6,
            num_lstm_layers=1,
        )
        self.fusers = nn.ModuleList(
            GatedFusion(enc, enc) for enc in self.encoder_channels
        )

    def forward(self, x: torch.Tensor, climate: torch.Tensor, **kwargs):
        size = x.size()[-2:]
        b = x.size(0)
        # Process each time step through the encoder.
        x_img = self._encode_images(x)
        # Encode climate data
        x_climate = self.climate_branch(climate)
        # Combine climate features with image features
        x_img = [rearrange(f, "b t c h w -> (b t) c h w") for f in x_img]
        x_climate = [rearrange(f, "b t c h w -> (b t) c h w") for f in x_climate]
        x = [
            fuser(i, c)  # Gated: fuser(i, c) # Conv: fuser(torch.cat([i, c], dim=2))
            for fuser, i, c in zip(self.fusers, x_img, x_climate)
        ]
        x = [rearrange(f, "(b t) c h w -> b t c h w", b=b) for f in x]
        # Pass the encoded sequence through the ConvLSTM.
        x = self._encode_timeseries(x)
        return self._decode(x, size=size)


class ClimateDemACTU(ClimateACTU):
    def __init__(
        self,
        in_channels,
        kernel_size,
        padding,
        stride,
        backbone,
        bias=True,
        batch_first=True,
        bidirectional=False,
        original_resolution=(256, 256),
        act_layer="sigmoid",
        seq_len=5,
        n_classes=1,
    ):
        super().__init__(
            in_channels + 1,
            kernel_size,
            padding,
            stride,
            backbone,
            bias,
            batch_first,
            bidirectional,
            original_resolution,
            act_layer,
            seq_len,
            n_classes,
        )

    def forward(self, x: torch.Tensor, climate: torch.Tensor, dem: torch.Tensor):
        dem = repeat(dem, "b c h w -> b t c h w", t=x.size(1))
        x = torch.cat([x, dem], dim=2)
        return super().forward(x, climate)


class DemACTU(ACTU):
    def __init__(
        self,
        in_channels,
        kernel_size,
        padding,
        stride,
        backbone,
        bias=True,
        batch_first=True,
        bidirectional=False,
        original_resolution=(256, 256),
        act_layer="sigmoid",
        n_classes=1,
    ):
        super().__init__(
            in_channels + 1,
            kernel_size,
            padding,
            stride,
            backbone,
            bias,
            batch_first,
            bidirectional,
            original_resolution,
            act_layer,
            n_classes,
        )

    def forward(self, x: torch.Tensor, dem: torch.Tensor, **kwargs):
        size = x.size()[-2:]
        dem = repeat(dem, "b c h w -> b t c h w", t=x.size(1))
        x = torch.cat([x, dem], dim=2)
        # Process each time step through the encoder.
        x = self._encode_images(x)
        # Pass the encoded sequence through the ConvLSTM.
        x = self._encode_timeseries(x)
        # x = self._fuse_dem(x, dem)
        return self._decode(x, size=size)


class ClimateBranchLSTM(nn.Module):
    """
    Processes climate time series data using an LSTM.
    Input shape: (B, T, T_1, C_clim) -> e.g., (B, 5, 6, 5)
    Output shape: (B, T, output_dim) -> e.g., (B, 5, 128)
    """

    def __init__(
        self,
        output_shapes: list[tuple[int, int, int]],
        climate_input_dim=5,
        climate_seq_len=6,
        lstm_hidden_dim=64,
        num_lstm_layers=1,
    ):
        super().__init__()
        self.climate_seq_len = climate_seq_len
        self.climate_input_dim = climate_input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.proj_dim = 128
        self.output_shapes = output_shapes

        self.lstm = nn.LSTM(
            input_size=climate_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,  # Crucial: expects input shape (batch, seq_len, features)
            dropout=0.3 if num_lstm_layers > 1 else 0,
            bidirectional=False,
        )

        # Linear layer to project LSTM output to the desired final dimension
        self.fc = nn.Linear(lstm_hidden_dim, self.proj_dim)

        self.upsamples = nn.ModuleList(
            _build_upsampler(self.proj_dim, *shape[:2]) for shape in output_shapes
        )

    def forward(self, climate_data: torch.Tensor) -> list[torch.Tensor]:
        # climate_data shape: (B, T, T_1, C_clim), e.g., (B, 5, 6, 5)
        B_img, B_cli, T, C = climate_data.shape

        # Reshape for LSTM: Treat each sequence independently
        lstm_input = rearrange(climate_data, "Bi Bc T C -> (Bi Bc) T C")

        # Pass through LSTM
        _, (hidden, _) = self.lstm.forward(lstm_input)
        # Get the last layer's hidden state
        last_hidden = (
            hidden[[hidden.size(0) // 2, -1]] if self.lstm.bidirectional else hidden[-1]
        )
        if last_hidden.ndim == 3:
            last_hidden = hidden.mean(dim=0)

        # Pass the final hidden state through the fully connected layer(s) and upsample
        climate_features = self.fc(last_hidden)
        climate_features = rearrange(climate_features, "b c -> b c 1 1")
        climate_features = [
            rearrange(
                u(climate_features), "(Bi Bc) C H W -> Bi Bc C H W", Bi=B_img, Bc=B_cli
            )
            for u in self.upsamples
        ]

        return climate_features


class GatedFusion(nn.Module):
    def __init__(self, img_channels, clim_channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(
                    img_channels + clim_channels, img_channels, kernel_size=3, padding=1
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(img_channels, img_channels, kernel_size=1),
                nn.Sigmoid(),  # Gate values between 0 and 1
            )
        )

    def forward(self, img_feat, clim_feat):
        gate = self.gate(torch.cat([img_feat, clim_feat], dim=1))
        return gate * img_feat + (1 - gate) * clim_feat


def _build_upsampler(
    in_channels: int, target_channels: int, target_h: int
) -> nn.Sequential:
    layers = []
    current_h = 1

    # Expand to target channels early (e.g., 1x1 → 1x1 with target_channels)
    layers += [nn.Conv2d(in_channels, target_channels, kernel_size=1), nn.GELU()]

    # Upsample spatially to target_h
    while current_h < target_h:
        next_h = min(current_h * 2, target_h)
        layers += [
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(target_channels, target_channels, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        current_h = next_h

    return nn.Sequential(*layers)
