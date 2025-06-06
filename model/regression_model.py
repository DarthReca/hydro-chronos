import matplotlib.pyplot as plt
import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import (
    MeanAbsoluteError,
    MeanSquaredError,
    MetricCollection,
    R2Score,
)

from .actu import ACTU
from .additional_actu import ClimateACTU, ClimateDemACTU, DemACTU
from .losses import MultiScaleLoss, WaveletLoss


class RegressionModel(LightningModule):
    def on_load_checkpoint(self, checkpoint):
        checkpoint["state_dict"] = {
            k: v for k, v in checkpoint["state_dict"].items() if "wavelet.dwt" not in k
        }
        return super().on_load_checkpoint(checkpoint)

    def __init__(
        self,
        target_length: int,
        lr: float,
        dem: bool = False,
        climate: str | bool = False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.target_length = target_length
        self.lr = lr

        if dem and climate:
            self.model = ClimateDemACTU(**kwargs)
        elif dem:
            self.model = DemACTU(**kwargs)
        elif climate:
            self.model = ClimateACTU(**kwargs)
        else:
            self.model = ACTU(**kwargs)

        self.multiscale = MultiScaleLoss(
            [0.75, 0.5, 0.25], nn.HuberLoss(reduction="none", delta=0.2)
        )
        self.wavelet = WaveletLoss(nn.HuberLoss(reduction="none", delta=0.2), "db2")

        self.loss_fn = lambda x, y: 0.5 * self.wavelet(x, y) + 0.5 * self.multiscale(
            x, y
        )

        self.regression_metric = MetricCollection(
            {
                "mae": MeanAbsoluteError(),
                "r2": R2Score(),
                "rmse": MeanSquaredError(squared=False),
            },
            prefix="regression_",
        )

        self.train_metrics = self.regression_metric.clone("train_")

    def forward(self, images, dem=None, climate=None, **kwargs):
        return self.model.forward(images, dem=dem, climate=climate)

    def training_step(self, batch, batch_idx):
        batch = {
            k: v.float() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }
        x, y = batch["images"], batch["mask"]
        y_hat = self.forward(**batch)
        loss = self._compute_loss(y_hat, y)
        self.log("train_loss", loss, batch_size=x.shape[0])
        self.train_metrics(y_hat.flatten(), y.flatten())
        self.log_dict(self.train_metrics, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        batch = {
            k: v.float() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }
        x, y = batch["images"], batch["mask"]
        y_hat = self.forward(**batch)

        loss = self._compute_loss(y_hat, y)
        self.regression_metric(y_hat.flatten(), y.flatten())
        # Logging
        self.log("val_loss", loss, batch_size=x.shape[0])
        self.log_dict(self.regression_metric, batch_size=x.shape[0])

        if batch_idx % 1 == 0 and self.logger is not None and y_hat[1] is not None:
            fig, axs = plt.subplots(1, 2)
            min_y, max_y = y.min(), y.max()
            # Select the most positive sample in batch
            idx = torch.argmax(y.flatten(start_dim=1).abs().sum(dim=1))
            cax0 = axs[0].imshow(
                y[idx].squeeze().cpu().numpy(), cmap="viridis", vmin=min_y, vmax=max_y
            )
            axs[0].set_title(f"True {batch['name'][idx]}")
            axs[1].imshow(
                y_hat[idx].squeeze().cpu().numpy(),
                cmap="viridis",
                vmin=min_y,
                vmax=max_y,
            )
            axs[1].set_title(f"Pred {batch['name'][idx]}")
            fig.colorbar(cax0, ax=axs, orientation="horizontal")
            self.logger.experiment.log_figure(f"val_images_{batch_idx}", fig)
            plt.close(fig)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.05,
            div_factor=1e6,
            final_div_factor=10,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def _compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor):
        loss = self.loss_fn(y_hat.squeeze(dim=1), y.squeeze(dim=1))
        if loss.dim() > 1:
            loss = loss.mean(dim=(1, 2))
        return loss.mean()
