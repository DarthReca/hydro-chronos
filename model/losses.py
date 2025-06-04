import numpy as np
import torch
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
from torch import nn


class MultiScaleLoss(nn.Module):
    def __init__(self, scales: list[float], loss_fn: nn.Module):
        super(MultiScaleLoss, self).__init__()
        assert all([s > 0 for s in scales]), "All scales must be positive"
        assert 1 not in scales, "Do not include the original scale in scales"
        self.scales = scales
        self.loss_fn = loss_fn

    def forward(self, prediction, target):
        """
        prediction: Tensor of shape [B, C, H, W]
        target: Tensor of shape [B, C, H, W]
        """
        if prediction.dim() != 4:
            prediction = prediction.unsqueeze(1)
        if target.dim() != 4:
            target = target.unsqueeze(1)

        total_loss = self.loss_fn(prediction, target).mean(dim=(1, 2, 3))
        for scale in self.scales:
            # Downsample using bilinear interpolation
            pred_scaled = F.interpolate(
                prediction, scale_factor=scale, mode="bilinear", align_corners=False
            )
            target_scaled = F.interpolate(
                target, scale_factor=scale, mode="bilinear", align_corners=False
            )
            # Compute the loss at this scale
            total_loss += self.loss_fn(pred_scaled, target_scaled).mean(dim=(1, 2, 3))

        # Return the average loss over scales
        return total_loss / len(self.scales)


class WaveletLoss(nn.Module):
    def __init__(self, loss_fn: nn.Module, wave="haar", mode="zero", levels=3):
        """
        Args:
            wave (str): Wavelet type, e.g., 'haar', 'db2', 'coif1', etc.
            mode (str): Padding mode, e.g., 'zero', 'symmetric', 'reflect'.
            levels (int): Number of decomposition levels in the wavelet transform.
        """
        super(WaveletLoss, self).__init__()
        self.dwt = DWTForward(J=levels, wave=wave, mode=mode)
        self.loss_fn = loss_fn

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predicted output, shape [B, C, H, W].
            target (torch.Tensor): Ground truth, shape [B, C, H, W].
        """
        if pred.dim() != 4:
            pred = pred.unsqueeze(1)
        if target.dim() != 4:
            target = target.unsqueeze(1)
        # Decompose both pred and target into lowpass and highpass coefficients
        # Yl_* are low-frequency (approximation) coefficients
        # Yh_* are high-frequency (detail) coefficients at each scale
        Yl_pred, Yh_pred = self.dwt(pred)
        Yl_targ, Yh_targ = self.dwt(target)

        # Compare the low-frequency approximation coefficients
        loss = self.loss_fn(Yl_pred, Yl_targ).mean(dim=(1, 2, 3)) * 0.5

        # Compare the high-frequency detail coefficients at each level
        for h_p, h_t, w in zip(
            Yh_pred, Yh_targ, np.linspace(1 / len(Yh_pred), 1, len(Yh_pred))
        ):
            # Yh_pred[lvl] has shape [B, C, 3, H_l, W_l] (horizontal, vertical, diagonal)
            loss += w * self.loss_fn(h_p, h_t).mean(dim=(1, 2, 3, 4))

        return loss
