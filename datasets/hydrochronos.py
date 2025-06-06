import math
import pathlib
import warnings
from dataclasses import dataclass
from datetime import date
from itertools import product
from typing import Literal

import albumentations as A
import h5py
import hdf5plugin
import numpy as np
import polars as pl
import psutil
import stocaching as st
import torch
import xarray as xr
from einops import rearrange, repeat
from scipy.special import expit
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.datasets import NonGeoDataset


@dataclass
class WaterData:
    full_input_timeseries: np.ndarray
    input_timeseries: np.ndarray
    output_timeseries: np.ndarray
    input_cloud_mask: np.ndarray
    output_cloud_mask: np.ndarray
    dem: np.ndarray
    climate: np.ndarray


CLIMATE_VARS = [
    "aet",
    "def",
    "pdsi",
    "pet",
    "pr",
    "ro",
    "soil",
    "srad",
    "swe",
    "tmmn",
    "tmmx",
    "vap",
    "vpd",
    "vs",
]


class HydroChronosDataModule(NonGeoDataModule):
    def __init__(self, cls, num_workers: int = 0, batch_size: int = 1, **kwargs):
        super().__init__(cls, num_workers=num_workers, batch_size=batch_size, **kwargs)
        self.aug = torch.nn.Identity()


class HydroChronos(NonGeoDataset):
    all_bands = ["B1", "B2", "B3", "B4", "B5", "B7"]
    rgb_bands = ["B3", "B2", "B1"]
    landsat_mean_std = (
        np.array([31.510, 29.306, 25.032, 62.795, 37.586, 21.682, 4.845]),
        np.array([18.089, 27.216, 27.928, 34.766, 26.263, 21.155, 0.848]),
    )
    sentinel_mean_std = (
        np.array([27.617, 24.782, 19.763, 60.604, 38.653, 21.459, 4.845]),
        np.array([20.668, 20.661, 23.357, 33.452, 25.807, 19.753, 0.848]),
    )
    classes_buckets = {2: [0.05], 3: [-0.05, 0.05]}

    def __init__(
        self,
        root: str,
        split: str,
        apply_augmentations: bool = True,
        keys_years: list[tuple[str, list[int], list[int]]] | None = None,
        h5_file: str = "water_landsat.h5",
        absolute_values: bool = False,
        climate_seq_len: int = 5,
        climate_bands: list[str] = ("tmmx", "pr", "ro", "soil", "aet"),
        classes: Literal[1, 2, 3] = 1,
    ):
        self.absolute_values = absolute_values
        self.h5_file = h5_file
        self.root = root
        self.split = split
        self.apply_augmentations = apply_augmentations and split == "train"
        self.class_threshold = (
            self.classes_buckets[classes] if classes in self.classes_buckets else None
        )

        self.train_augs = A.Compose(
            [
                A.D4(),
                A.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, p=1.0),
                A.RandomResizedCrop(size=(256, 256), scale=(0.5, 1.0)),
            ],
            additional_targets={"dem": "image"},
        )
        # Load splits
        self.input_output_years = self._load_temporal_split()
        # Load all files name
        with h5py.File(f"{root}/{h5_file}", "r") as f:
            keys_years = keys_years or map(
                lambda k: (k[0], *k[1]),
                product(f.keys(), self.input_output_years),
            )
            self.files = [
                (k, np.array(in_years), np.array(out_years))
                for k, in_years, out_years in keys_years
                if (
                    k in f
                    and len(np.intersect1d(in_years, f[k].attrs["years"]))
                    >= 0.8 * len(in_years)
                )
            ]
        with h5py.File("data/climate.h5") as f:
            keys = set(f.keys())
        self.files = [k for k in self.files if k[0] in keys]
        self.selected_bands = climate_bands
        self.seq_len = climate_seq_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        sample = self._load_data(*self.files[idx])
        # Standardize the images
        if self.h5_file == "water_landsat.h5":
            mean, std = self.landsat_mean_std
        elif self.h5_file == "water_sentinel.h5":
            mean, std = self.sentinel_mean_std
        else:
            raise ValueError(f"Unknown h5 file: {self.h5_file}")
        time_len = sample["images"].size(0)
        mean = repeat(mean, "c -> b c h w", b=time_len, h=256, w=256)
        std = repeat(std, "c -> b c h w", b=time_len, h=256, w=256)
        sample["images"] = (sample["images"].float() - mean) / std
        if self.class_threshold is not None:
            sample["mask"] = torch.bucketize(
                sample["mask"].float(), self.class_threshold
            )
        sample |= {
            "name": self.files[idx][0].split("/")[-1],
            "in_years": self.files[idx][1],
            "out_years": self.files[idx][2],
        }
        # Apply augmentations
        if self.apply_augmentations:
            sample = self._apply_augmentations(**sample)
        return sample

    def _load_data(self, key: str, in_years: np.ndarray, out_years: np.ndarray):
        data = self._load_time_series(key, in_years, out_years)
        input_image = data.full_input_timeseries.transpose(1, 0, 2, 3)
        diff_mndwi = self._compute_mask(data)[np.newaxis]
        # Convert the inputs to a PyTorch tensor
        return {
            "images": torch.from_numpy(input_image),
            "mask": torch.from_numpy(diff_mndwi),
            "dem": torch.from_numpy(data.dem),
        }

    def _apply_augmentations(self, **kwargs: torch.Tensor | str):
        to_transform = {k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor)}
        for k, v in list(to_transform.items()):
            if v.ndim == 4:
                to_transform[k] = rearrange(v, "b c h w -> b h w c").numpy()
            elif v.ndim == 3:
                to_transform[k] = rearrange(v, "c h w -> h w c").numpy()
        # Get the parameters for the augmentations
        to_transform = self.train_augs(**to_transform)
        # Convert the inputs to a PyTorch tensor
        for k, v in list(to_transform.items()):
            if v.ndim == 3:
                to_transform[k] = torch.from_numpy(rearrange(v, "h w c -> c h w"))
            elif v.ndim == 4:
                to_transform[k] = torch.from_numpy(rearrange(v, "b h w c -> b c h w"))
        return kwargs | to_transform

    def _compute_mndwi(self, ds: np.ndarray) -> np.ndarray:
        ds = ds.astype("float32")
        b5 = self.all_bands.index("B5")
        b2 = self.all_bands.index("B2")
        return (ds[b2] - ds[b5]) / (ds[b2] + ds[b5] + 1e-9)

    def _load_time_series(
        self, key: str, in_years: np.ndarray, out_years: np.ndarray
    ) -> WaterData:
        with h5py.File(f"{self.root}/{self.h5_file}", "r") as f:
            bands = f[key]["bands"]
            cloud_mask = f[key]["qa_mask"][...]
            dem = f[key]["dem"][...].astype("float32")
            years = f[key].attrs["years"]
        dem = np.log(np.clip(dem + 30, 1e-6, None))
        if self.h5_file == "water_landsat.h5":
            bands = bands.astype("uint8")
        elif self.h5_file == "water_sentinel.h5":
            bands = (bands.astype("float32") / 10000) * 255
            bands = bands.astype("uint8")

        all_input_years = [in_y for in_y, _ in self.input_output_years]
        all_input_years = max(
            all_input_years, key=lambda x: np.intersect1d(in_years, x).size
        )
        # Create a mask for the invalid pixels
        relevant_bands = [self.all_bands.index("B5"), self.all_bands.index("B2")]
        no_data_mask = (bands[relevant_bands] == 0).any(axis=0)
        unclear_mask = cloud_mask | no_data_mask
        input_cloud_mask = unclear_mask[in_years]
        output_cloud_mask = unclear_mask[out_years]
        # Get the input and output images
        in_years = (in_years.min() <= years) & (years <= in_years.max())
        out_years = (out_years.min() <= years) & (years <= out_years.max())
        in_series = bands[:, in_years]
        out_series = bands[:, out_years]
        # Zero impute the missing input years
        input_image = np.zeros(
            (in_series.shape[0], len(all_input_years), *in_series.shape[2:]),
            dtype=in_series.dtype,
        )
        for i, year in enumerate(all_input_years):
            if year in years:
                input_image[:, i] = bands[:, years == year].squeeze()
        # Load the climate data
        climate = self._load_climate_data(key, years, in_years)
        return WaterData(
            input_image,
            in_series,
            out_series,
            input_cloud_mask,
            output_cloud_mask,
            dem,
            climate,
        )

    def _load_temporal_split(self):
        if self.h5_file == "water_sentinel.h5":
            return [(np.arange(2015, 2020), np.arange(2020, 2025))]
        # Landsat data is split into train and validation sets.
        if self.split == "train":
            return [
                (np.arange(1990, 1995), np.arange(1995, 2000)),
                (np.arange(1995, 2000), np.arange(2000, 2005)),
            ]
        elif self.split == "val":
            return [(np.arange(2000, 2005), np.arange(2005, 2010))]
        else:
            raise ValueError("Landsat should not be used for testing.")

    def _compute_mask(self, data: WaterData) -> np.ndarray:
        in_mndwi = self._compute_mndwi(data.input_timeseries)
        out_mndwi = self._compute_mndwi(data.output_timeseries)
        # Remove clouds and shadows
        in_mndwi[data.input_cloud_mask] = np.nan
        out_mndwi[data.output_cloud_mask] = np.nan
        # Compute the difference in MNDWI
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        in_median = np.nanmedian(in_mndwi, axis=0)
        out_median = np.nanmedian(out_mndwi, axis=0)
        diff_mndwi = in_median - out_median
        warnings.filterwarnings("default", message="All-NaN slice encountered")
        # Fill NaN values
        diff_mndwi = np.nan_to_num(diff_mndwi, nan=0)
        # Normalize the mask
        if self.absolute_values:
            diff_mndwi = abs(diff_mndwi)
        return diff_mndwi / 2

    def _load_climate_data(self, name: str, end_dates, in_years) -> np.ndarray:
        with h5py.File("data/climate.h5") as f:
            climate_dates = [date.fromisoformat(str(t)) for t in f[name]["time"][...]]
            end_indices = [climate_dates.index(d) for d in end_dates]
            start_indices = [max(i - self.seq_len, 0) for i in end_indices]
            climate_indices = np.zeros(f[name]["climate"].shape[1], dtype=bool)
            for start, end in zip(start_indices, end_indices):
                climate_indices[start:end] = True
            climate_bands = f[name]["climate"][:, climate_indices].astype("float32")
            if climate_bands.shape[1] != len(in_years) * self.seq_len:
                climate_bands = np.concatenate(
                    [climate_bands, np.ones((climate_bands.shape[0], 1)) * np.nan],
                    axis=1,
                )
        # Standardize climate bands
        climate_bands = standardize_climate_var(climate_bands)
        # Select bands
        indexes = [CLIMATE_VARS.index(k) for k in self.selected_bands]
        climate_bands = climate_bands[indexes]
        climate_bands = np.concatenate(
            [climate_bands, np.isnan(climate_bands).any(axis=0)[np.newaxis]]
        )
        np.nan_to_num(climate_bands, copy=False)
        return rearrange(climate_bands, "c (b t) -> b t c", b=self.seq_len)


def standardize_climate_var(climate_bands: np.ndarray) -> np.ndarray:
    # Fill missing values
    climate_bands[climate_bands == -32768] = np.nan
    # Precipitation bands (log1p + Standard scaling)
    index = CLIMATE_VARS.index("pr")
    climate_bands[index] = (np.log1p(climate_bands[index]) - 4.16) / 1.05
    # Max temp band (Standard scaling)
    index = CLIMATE_VARS.index("tmmx")
    climate_bands[index] = (climate_bands[index] - 192.2) / 120.0
    # Min temp band (Standard scaling)
    index = CLIMATE_VARS.index("tmmn")
    climate_bands[index] = (climate_bands[index] - 86.2) / 111.15
    # Actual Evapotranspiration band (log1p + Standard scaling)
    index = CLIMATE_VARS.index("aet")
    climate_bands[index] = (np.log1p(climate_bands[index]) - 5.45) / 2.25
    # Runoff band (log1p + Standard scaling)
    index = CLIMATE_VARS.index("ro")
    climate_bands[index] = (np.log1p(climate_bands[index]) - 2.19) / 1.68
    # Soil moisture band (log1p + Standard scaling)
    index = CLIMATE_VARS.index("soil")
    climate_bands[index] = (np.log1p(climate_bands[index]) - 6.42) / 1.09
    return climate_bands
