import matplotlib.pyplot as plt
import torch
from lightning import LightningModule
from monai.losses.dice import GeneralizedDiceFocalLoss
from torchmetrics import ClasswiseWrapper, F1Score

from .actu import ACTU
from .additional_actu import ClimateACTU, ClimateDemACTU, DemACTU


class ClassificationModel(LightningModule):
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

        assert kwargs["n_classes"] != 1, (
            "n_classes should be greater than 1 for classification"
        )
        if dem and climate:
            self.model = ClimateDemACTU(**kwargs)
        elif dem:
            self.model = DemACTU(**kwargs)
        elif climate:
            self.model = ClimateACTU(**kwargs)
        else:
            self.model = ACTU(**kwargs)
        self.loss_fn = GeneralizedDiceFocalLoss(softmax=True, to_onehot_y=True)
        self.image_metrics = ClasswiseWrapper(
            F1Score("multiclass", num_classes=kwargs["n_classes"], average=None),
            prefix="f1_",
        )

    def forward(self, images, dem=None, climate=None, **kwargs):
        return self.model.forward(images, dem=dem, climate=climate)

    def training_step(self, batch, batch_idx):
        batch = {
            k: v.float() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }
        x, y = batch["images"], batch["mask"].long()
        y_hat = self.forward(**batch)
        loss = self._compute_loss(y_hat, y.squeeze(dim=1))
        self.log("train_loss", loss, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        batch = {
            k: v.float() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }
        x, y = batch["images"], batch["mask"].long()
        y_hat = self.forward(**batch)

        loss = self._compute_loss(y_hat, y)
        self.image_metrics.update(y_hat, y.squeeze(dim=1))
        # Logging
        self.log("val_loss", loss, batch_size=x.shape[0])

        if batch_idx % 1 == 0 and self.logger is not None and y_hat[1] is not None:
            fig, axs = plt.subplots(1, 2)
            # Select the most positive sample in batch
            idx = torch.argmax(y.flatten(start_dim=1).abs().sum(dim=1))
            cax0 = axs[0].imshow(
                y[idx].squeeze().cpu().numpy(),
                vmin=0,
                vmax=self.model.n_classes - 1,
            )
            axs[0].set_title(f"True {batch['name'][idx]}")
            axs[1].imshow(
                y_hat[idx].argmax(0).squeeze().cpu().numpy(),
                vmin=0,
                vmax=self.model.n_classes - 1,
            )
            axs[1].set_title(f"Pred {batch['name'][idx]}")
            fig.colorbar(cax0, ax=axs, orientation="horizontal")
            self.logger.experiment.log_figure(f"val_images_{batch_idx}", fig)
            plt.close(fig)

        return loss

    def on_validation_epoch_end(self):
        self.log_dict(self.image_metrics.compute())
        self.image_metrics.reset()

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
        if y.dim() != 4:
            y = y.unsqueeze(1)
        loss = self.loss_fn(y_hat.squeeze(dim=1), y)
        if loss.dim() > 1:
            loss = loss.mean(dim=(1, 2))
        return loss.mean()
