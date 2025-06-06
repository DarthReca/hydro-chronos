import polars as pl
from torchgeo.datasets import NonGeoDataset

from .hydrochronos import HydroChronos


class PretrainWaterDataset(NonGeoDataset):
    def __init__(self, root, split, thereshold=0.05, **kwargs):
        self.threshold = thereshold

        keys = pl.read_parquet(f"{root}/landsat_samples.parquet")
        keys = keys.filter(split=split)
        keys = keys.select("key", "in_years", "out_years").rows()

        self.dataset = HydroChronos(
            root,
            split,
            keys_years=keys,
            h5_file="water_landsat.h5",
            **kwargs,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sample["mask"][abs(sample["mask"]) < self.threshold] = 0
        assert ((abs(sample["mask"]) > self.threshold) | (sample["mask"] == 0)).all()
        return sample


class SentinelWaterDataset(NonGeoDataset):
    def __init__(self, root, split, **kwargs):
        keys = pl.read_parquet(f"{root}/sentinel_samples.parquet")
        keys = keys.filter(split=split)
        keys = keys.select("key", "in_years", "out_years").rows()

        self.dataset = HydroChronos(
            root, split, keys_years=keys, h5_file="water_sentinel.h5", **kwargs
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
