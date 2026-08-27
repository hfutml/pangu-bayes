import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import netCDF4 as nc
import pandas as pd
import json
from datetime import datetime, timedelta

class WeatherDataset40(Dataset):
    def __init__(self, patch_size, is_train=False):
        super().__init__()
        self.is_train = is_train
        self.patch_size = patch_size
        self.file_pre = "train" if is_train else "test"
        dir_file = os.listdir(f'../ERA5_examples/{self.file_pre}/')

        if self.file_pre == "test":
            dir_file = [f for f in dir_file if f.startswith('2018')]
        self.dir_files = sorted(dir_file)
        self.idx_len = int(len(dir_file))
        self.mean_surface = torch.load(('../ERA5_examples/statistic/surface_mean_40.pt'), weights_only=False).unsqueeze(-1).unsqueeze(-1)
        self.mean_upper = torch.load(('../ERA5_examples/statistic/upper_mean_40.pt'), weights_only=False).unsqueeze(-1).unsqueeze(-1)
        self.std_surface = torch.load(('../ERA5_examples/statistic/surface_std_40.pt'), weights_only=False).unsqueeze(-1).unsqueeze(-1)
        self.std_upper = torch.load(('../ERA5_examples/statistic/upper_std_40.pt'), weights_only=False).unsqueeze(-1).unsqueeze(-1)
    
    def normlize(self, upper, surface):
        upper = (upper - self.mean_upper)/ self.std_upper
        surface  = (surface - self.mean_surface)/ self.std_surface 
        return upper, surface

    def __len__(self):
        return self.idx_len - 2

    def __getitem__(self, idx):
        idx0 = self.dir_files[idx]
        idx1 = self.dir_files[idx + 1]
        idx2 = self.dir_files[idx + 2]


        input_dict = torch.load(f'../ERA5_examples/{self.file_pre}/{idx0}')
        input_upper, input_surface = input_dict['input_upper'], input_dict['input_surface']
        
        input_upper, input_surface = self.normlize(input_upper, input_surface)

        input_dict = torch.load(f'../ERA5_examples/{self.file_pre}/{idx1}')
        input_upper1, input_surface1 = input_dict['input_upper'], input_dict['input_surface']

        input_upper1, input_surface1 = self.normlize(input_upper1, input_surface1)


        input_upper, input_surface = torch.cat([input_upper, input_upper1], dim=0), torch.cat([input_surface, input_surface1], dim=0)
        

        target_dict = torch.load(f'../ERA5_examples/{self.file_pre}/{idx2}')
        target_upper, target_surface = target_dict['input_upper'], target_dict['input_surface']

        target_upper, target_surface = self.normlize(target_upper, target_surface)



        C, Z, H, W = input_upper.shape
        if W % self.patch_size[2] != 0:
            input_upper = F.pad(input_upper, (0, self.patch_size[2] - W % self.patch_size[2]))
            input_surface = F.pad(input_surface, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            input_upper = F.pad(input_upper, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
            input_surface = F.pad(input_surface, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if Z % self.patch_size[0] != 0:
            input_upper = F.pad(input_upper, (0, 0, 0, 0, 0, self.patch_size[0] - Z % self.patch_size[0]))

        return input_upper, input_surface, target_upper, target_surface


class WeatherDataset40test(Dataset):
    def __init__(self, patch_size, steps, is_train=False):
        super().__init__()
        self.is_train = is_train
        self.patch_size = patch_size
        self.steps = steps
        self.file_pre = "train" if is_train else "test"
        dir_file = os.listdir(f'../ERA5_examples/{self.file_pre}/')

        if self.file_pre == "test":
            dir_file = [f for f in dir_file if f.startswith('2022')]
        self.dir_files = sorted(dir_file)
        self.idx_len = int(len(dir_file))
        self.mean_surface = torch.load(('../ERA5_examples/statistic/surface_mean_40.pt'), weights_only=False).unsqueeze(-1).unsqueeze(-1)
        self.mean_upper = torch.load(('../ERA5_examples/statistic/upper_mean_40.pt'), weights_only=False).unsqueeze(-1).unsqueeze(-1)
        self.std_surface = torch.load(('../ERA5_examples/statistic/surface_std_40.pt'), weights_only=False).unsqueeze(-1).unsqueeze(-1)
        self.std_upper = torch.load(('../ERA5_examples/statistic/upper_std_40.pt'), weights_only=False).unsqueeze(-1).unsqueeze(-1)

    def normlize(self, upper, surface):
        upper = (upper - self.mean_upper)/ self.std_upper
        surface  = (surface - self.mean_surface)/ self.std_surface 
        return upper, surface

    def __len__(self):
        return (self.idx_len - self.steps - 1) // 2

    def __getitem__(self, idx):
        file_idx = idx * 2 + 1
        idx0 = self.dir_files[file_idx]
        idx1 = self.dir_files[file_idx + 1]
        idx2 = self.dir_files[file_idx + 1 + self.steps]
        input_dict = torch.load(f'../ERA5_examples/{self.file_pre}/{idx0}')
        input_upper, input_surface = input_dict['input_upper'], input_dict['input_surface']
        input_upper, input_surface = self.normlize(input_upper, input_surface)

        
        input_dict = torch.load(f'../ERA5_examples/{self.file_pre}/{idx1}')
        input_upper1, input_surface1 = input_dict['input_upper'], input_dict['input_surface']
        input_upper1, input_surface1 = self.normlize(input_upper1, input_surface1)

        input_upper, input_surface = torch.cat([input_upper, input_upper1], dim=0), torch.cat([input_surface, input_surface1], dim=0)


        target_dict = torch.load(f'../ERA5_examples/{self.file_pre}/{idx2}')
        target_upper, target_surface = target_dict['input_upper'], target_dict['input_surface']


        C, Z, H, W = input_upper.shape
        if W % self.patch_size[2] != 0:
            input_upper = F.pad(input_upper, (0, self.patch_size[2] - W % self.patch_size[2]))
            input_surface = F.pad(input_surface, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            input_upper = F.pad(input_upper, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
            input_surface = F.pad(input_surface, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if Z % self.patch_size[0] != 0:
            input_upper = F.pad(input_upper, (0, 0, 0, 0, 0, self.patch_size[0] - Z % self.patch_size[0]))

        return input_upper, input_surface, target_upper, target_surface


class WeatherDataset40buff(Dataset):
    def __init__(self, patch_size, is_train=False):
        super().__init__()
        self.is_train = is_train
        self.patch_size = patch_size
        self.file_pre = "train" if is_train else "test"
        base_dir = f'../ERA5_examples/{self.file_pre}/'
        dir_file = os.listdir(base_dir)

        if self.file_pre == "test":
            dir_file = [f for f in dir_file if f.startswith('2018')]
        
        self.dir_files = sorted(dir_file)
        self.idx_len = int(len(dir_file))

        self.mean_surface = torch.load('../ERA5_examples/statistic/surface_mean_40.pt', weights_only=False).unsqueeze(-1).unsqueeze(-1)
        self.mean_upper = torch.load('../ERA5_examples/statistic/upper_mean_40.pt', weights_only=False).unsqueeze(-1).unsqueeze(-1)
        self.std_surface = torch.load('../ERA5_examples/statistic/surface_std_40.pt', weights_only=False).unsqueeze(-1).unsqueeze(-1)
        self.std_upper = torch.load('../ERA5_examples/statistic/upper_std_40.pt', weights_only=False).unsqueeze(-1).unsqueeze(-1)
    
    def normlize(self, upper, surface):
        upper = (upper - self.mean_upper) / self.std_upper
        surface  = (surface - self.mean_surface) / self.std_surface 
        return upper, surface


    def __len__(self):
        return self.idx_len - 2

    def get_target(self, idx):

        file_idx = self.dir_files[idx + 2] 
        input_dict = torch.load(f'../ERA5_examples/{self.file_pre}/{file_idx}')
        
        target_upper, target_surface = input_dict['input_upper'], input_dict['input_surface']
  
        target_upper, target_surface = self.normlize(target_upper, target_surface)
        
        return target_upper, target_surface

    def __getitem__(self, idx):
        idx0 = self.dir_files[idx]
        idx1 = self.dir_files[idx + 1]
        idx2 = self.dir_files[idx + 2]

        input_dict0 = torch.load(f'../ERA5_examples/{self.file_pre}/{idx0}')
        u0, s0 = self.normlize(input_dict0['input_upper'], input_dict0['input_surface'])
        input_dict1 = torch.load(f'../ERA5_examples/{self.file_pre}/{idx1}')
        u1, s1 = self.normlize(input_dict1['input_upper'], input_dict1['input_surface'])
        input_upper = torch.cat([u0, u1], dim=0)
        input_surface = torch.cat([s0, s1], dim=0)
        
        # T+1 (Target)
        input_dict2 = torch.load(f'../ERA5_examples/{self.file_pre}/{idx2}')
        target_upper, target_surface = self.normlize(input_dict2['input_upper'], input_dict2['input_surface'])

        return input_upper, input_surface, u1, s1, target_upper, target_surface, idx


import os

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


class WeatherDataset40hresbuff(Dataset):
    """
    HRES-analysis training dataset.

    Each sample contains:
      input:  HRES[t], HRES[t+1]
      target: HRES[t+2]

    get_target(sample_id + s) returns HRES[t+2+s], which preserves the
    interface used by the existing autoregressive training loop.
    """

    _YEAR_INDEX_STRIDE = 10_000_000

    def __init__(
        self,
        patch_size,
        is_train=False,
        max_ar_steps=4,
        hres_root="/home",
        years=range(2016, 2022),
        statistic_root="../ERA5_examples/statistic",
    ):
        super().__init__()
        self.is_train = is_train
        self.patch_size = patch_size
        self.max_ar_steps = max_ar_steps
        self.hres_root = hres_root
        self.years = list(years)
        self._hres = {}

        if self.max_ar_steps < 1:
            raise ValueError("max_ar_steps must be at least 1")

        self.hres_paths = [
            os.path.join(
                self.hres_root,
                (
                    f"hres_t0_{year}_zqtuv_t2m_10u_10v_"
                    "msl_13levels.zarr"
                ),
            )
            for year in self.years
        ]

        self.upper_vars = ["z", "q", "t", "u", "v"]
        self.surface_vars = ["t2m", "10u", "10v", "msl"]

        self.mean_surface = torch.load(
            os.path.join(statistic_root, "surface_mean_40.pt"),
            weights_only=False,
        ).float().unsqueeze(-1).unsqueeze(-1)

        self.mean_upper = torch.load(
            os.path.join(statistic_root, "upper_mean_40.pt"),
            weights_only=False,
        ).float().unsqueeze(-1).unsqueeze(-1)

        self.std_surface = torch.load(
            os.path.join(statistic_root, "surface_std_40.pt"),
            weights_only=False,
        ).float().unsqueeze(-1).unsqueeze(-1)

        self.std_upper = torch.load(
            os.path.join(statistic_root, "upper_std_40.pt"),
            weights_only=False,
        ).float().unsqueeze(-1).unsqueeze(-1)

        # Each item is (year_index, start_time_index). Samples never cross
        # year boundaries and contain enough targets for max_ar_steps.
        self.samples = []
        self.time_sizes = []

        for year_index, path in enumerate(self.hres_paths):
            if not os.path.exists(path):
                raise FileNotFoundError(f"HRES zarr not found: {path}")

            ds = xr.open_zarr(path, consolidated=False)
            try:
                if "time" not in ds.sizes:
                    raise ValueError(
                        f"No time dimension in {path}: {dict(ds.sizes)}"
                    )
                time_size = int(ds.sizes["time"])
            finally:
                ds.close()

            self.time_sizes.append(time_size)

            # Inputs use t and t+1. AR K targets use t+2 through t+K+1.
            number_of_starts = max(
                0,
                time_size - self.max_ar_steps - 1,
            )
            self.samples.extend(
                (year_index, start_idx)
                for start_idx in range(number_of_starts)
            )

        if not self.samples:
            raise ValueError("No valid HRES training samples were found")

    def _open_hres(self, year_index):
        if year_index not in self._hres:
            self._hres[year_index] = xr.open_zarr(
                self.hres_paths[year_index],
                consolidated=False,
            )
        return self._hres[year_index]

    @staticmethod
    def _get_var(ds, names):
        for name in names:
            if name in ds.data_vars:
                return ds[name]
        raise KeyError(
            f"Cannot find any of {names}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    @staticmethod
    def _find_dim(da, candidates, variable_name):
        for name in candidates:
            if name in da.dims:
                return name
        raise ValueError(
            f"Cannot identify {variable_name} dimension in {da.dims}"
        )

    def _prepare_upper_array(self, da, time_idx):
        da = da.isel(time=time_idx).squeeze(drop=True)

        level_dim = self._find_dim(
            da,
            ["level", "pressure_level", "isobaricInhPa"],
            "pressure-level",
        )
        lat_dim = self._find_dim(
            da,
            ["latitude", "lat"],
            "latitude",
        )
        lon_dim = self._find_dim(
            da,
            ["longitude", "lon"],
            "longitude",
        )

        da = da.transpose(level_dim, lat_dim, lon_dim)
        arr = np.asarray(da.values, dtype=np.float32)

        # The model expects latitude ordered north to south.
        lat = np.asarray(da[lat_dim].values)
        if lat[0] < lat[-1]:
            arr = arr[:, ::-1, :].copy()

        return arr

    def _prepare_surface_array(self, da, time_idx):
        da = da.isel(time=time_idx).squeeze(drop=True)

        lat_dim = self._find_dim(
            da,
            ["latitude", "lat"],
            "latitude",
        )
        lon_dim = self._find_dim(
            da,
            ["longitude", "lon"],
            "longitude",
        )

        da = da.transpose(lat_dim, lon_dim)
        arr = np.asarray(da.values, dtype=np.float32)

        # The model expects latitude ordered north to south.
        lat = np.asarray(da[lat_dim].values)
        if lat[0] < lat[-1]:
            arr = arr[::-1, :].copy()

        return arr

    def _read_hres_upper_surface(self, year_index, time_idx):
        ds = self._open_hres(year_index)

        upper_list = []
        for variable_name in self.upper_vars:
            da = self._get_var(ds, [variable_name])
            arr = self._prepare_upper_array(da, time_idx)
            upper_list.append(torch.from_numpy(arr))

        surface_aliases = {
            "t2m": ["t2m", "2t"],
            "10u": ["10u", "u10"],
            "10v": ["10v", "v10"],
            "msl": ["msl", "mslp"],
        }

        surface_list = []
        for variable_name in self.surface_vars:
            da = self._get_var(
                ds,
                surface_aliases.get(variable_name, [variable_name]),
            )
            arr = self._prepare_surface_array(da, time_idx)
            surface_list.append(torch.from_numpy(arr))

        upper = torch.stack(upper_list, dim=0)
        surface = torch.stack(surface_list, dim=0)

        expected_upper = tuple(self.mean_upper.shape[:2])
        expected_surface = (self.mean_surface.shape[0],)

        if upper.shape[:2] != expected_upper:
            raise ValueError(
                "HRES upper-air shape does not match the statistics: "
                f"data={tuple(upper.shape)}, "
                f"statistics={tuple(self.mean_upper.shape)}"
            )
        if surface.shape[:1] != expected_surface:
            raise ValueError(
                "HRES surface shape does not match the statistics: "
                f"data={tuple(surface.shape)}, "
                f"statistics={tuple(self.mean_surface.shape)}"
            )

        return upper, surface

    def normlize(self, upper, surface):
        upper = (upper.float() - self.mean_upper) / self.std_upper
        surface = (
            surface.float() - self.mean_surface
        ) / self.std_surface
        return upper, surface

    def __len__(self):
        return len(self.samples)

    def _encode_sample_id(self, year_index, start_idx):
        return year_index * self._YEAR_INDEX_STRIDE + start_idx

    def _decode_sample_id(self, sample_id):
        sample_id = int(sample_id)
        year_index, start_idx = divmod(
            sample_id,
            self._YEAR_INDEX_STRIDE,
        )

        if not 0 <= year_index < len(self.hres_paths):
            raise IndexError(f"Invalid encoded sample ID: {sample_id}")

        return year_index, start_idx

    def get_target(self, sample_id):
        """
        Return the target associated with an encoded sample ID.

        If the training loop calls get_target(base_id + s), the returned
        target is HRES[t+2+s] from the same year.
        """
        year_index, start_idx = self._decode_sample_id(sample_id)
        target_time_idx = start_idx + 2

        if target_time_idx >= self.time_sizes[year_index]:
            raise IndexError(
                "Requested target exceeds the available HRES times: "
                f"year={self.years[year_index]}, "
                f"time_idx={target_time_idx}"
            )

        target_upper, target_surface = (
            self._read_hres_upper_surface(
                year_index,
                target_time_idx,
            )
        )
        return self.normlize(target_upper, target_surface)

    def __getitem__(self, idx):
        year_index, start_idx = self.samples[idx]

        upper_0, surface_0 = self._read_hres_upper_surface(
            year_index,
            start_idx,
        )
        upper_1, surface_1 = self._read_hres_upper_surface(
            year_index,
            start_idx + 1,
        )
        target_upper, target_surface = (
            self._read_hres_upper_surface(
                year_index,
                start_idx + 2,
            )
        )

        upper_0, surface_0 = self.normlize(upper_0, surface_0)
        upper_1, surface_1 = self.normlize(upper_1, surface_1)
        target_upper, target_surface = self.normlize(
            target_upper,
            target_surface,
        )

        input_upper = torch.cat([upper_0, upper_1], dim=0)
        input_surface = torch.cat([surface_0, surface_1], dim=0)

        sample_id = self._encode_sample_id(
            year_index,
            start_idx,
        )

        return (
            input_upper,
            input_surface,
            upper_1,
            surface_1,
            target_upper,
            target_surface,
            sample_id,
        )