import time

import comet_ml
import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CometLogger
from omegaconf import DictConfig, OmegaConf

import datasets
from model import ClassificationModel


@hydra.main(version_base="1.3", config_path="configs", config_name="class_train")
def main(args: DictConfig):
    args = OmegaConf.to_container(args, resolve=True)
    L.seed_everything(args["seed"])
    torch.set_float32_matmul_precision("high")
    if "checkpoint" in args and args["checkpoint"] is not None:
        print("Loading from checkpoint")
        model = ClassificationModel.load_from_checkpoint(
            args["checkpoint"], map_location="cpu", lr=args["model"]["lr"], strict=False
        )
    else:
        model = ClassificationModel(**args["model"])

    dm = datasets.HydroChronosDataModule(
        datasets.SentinelWaterDataset
        if "checkpoint" in args
        else datasets.PretrainWaterDataset,
        **args["dataset"],
    )
    experiment_id = time.strftime("%Y%m%d-%H%M%S")
    logger = False
    if args["logger"] == "comet":
        logger = CometLogger(save_dir="comet-logs", offline=False)
        experiment_id = logger.experiment.id
    if logger and "checkpoint" in args:
        logger.log_hyperparams({"checkpoint": args["checkpoint"]})
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            monitor="val_loss",
            dirpath=f"checkpoints/{experiment_id}",
            filename="waterpredictor-{epoch:02d}-{val_loss:.4f}",
            save_top_k=2,
            save_last=True,
            mode="min",
            save_weights_only=True,
        ),
    ]

    trainer = L.Trainer(
        **args["trainer"],
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=0.5,
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
