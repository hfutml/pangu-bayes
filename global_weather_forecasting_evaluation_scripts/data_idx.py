import os
import torch
import numpy as np
import xarray as xr
from torch.utils.data import Dataset


class WeatherDataset40test(Dataset):
    def __init__(
        self,
        patch_size,
        steps,
        is_train=False,
        era5_root="/ERA5_40_idx",
        hres_zarr_path="/hres_t0_2022_zqtuv_t2m_10u_10v_msl_13levels.zarr",
    ):
        super().__init__()
        self.is_train = is_train
        self.patch_size = patch_size
        self.steps = steps
        self.file_pre = "train" if is_train else "test"

        self.era5_root = era5_root
        self.hres_zarr_path = hres_zarr_path
        self._hres = None

        dir_file = os.listdir(f"{self.era5_root}/{self.file_pre}/")
        if self.file_pre == "test":
            dir_file = [f for f in dir_file if f.startswith("2022")]

        self.dir_files = sorted(dir_file)
        self.idx_len = len(self.dir_files)

        self.upper_vars = ["z", "q", "t", "u", "v"]
        self.surface_vars = ["t2m", "10u", "10v", "msl"]

        self.mean_surface = torch.load(
            "../ERA5_examples/statistic/surface_mean_40.pt",
            weights_only=False
        ).unsqueeze(-1).unsqueeze(-1)

        self.mean_upper = torch.load(
            "../ERA5_examples/statistic/upper_mean_40.pt",
            weights_only=False
        ).unsqueeze(-1).unsqueeze(-1)

        self.std_surface = torch.load(
            "../ERA5_examples/statistic/surface_std_40.pt",
            weights_only=False
        ).unsqueeze(-1).unsqueeze(-1)

        self.std_upper = torch.load(
            "../ERA5_examples/statistic/upper_std_40.pt",
            weights_only=False
        ).unsqueeze(-1).unsqueeze(-1)

    def _open_hres(self):
        if self._hres is None:
            self._hres = xr.open_zarr(self.hres_zarr_path, consolidated=False)
        return self._hres

    def _get_var(self, ds, names):
        for name in names:
            if name in ds:
                return ds[name]
        raise KeyError(
            f"Cannot find any of variables {names} in zarr. "
            f"Available variables: {list(ds.data_vars)}"
        )

    def _read_hres_upper_surface(self, time_idx):
        ds = self._open_hres()

        upper_list = []
        for var_name in self.upper_vars:
            da = self._get_var(ds, [var_name])

            arr = da.isel(time=time_idx).values.astype(np.float32)
            arr = np.asarray(arr)
            arr = np.squeeze(arr)

            # HRES latitude: -90 -> 90
            # ERA5/model expected latitude: 90 -> -90
            arr = arr[..., ::-1, :].copy()

            upper_list.append(torch.from_numpy(arr))

        surface_aliases = {
            "t2m": ["t2m", "2t"],
            "10u": ["10u", "u10"],
            "10v": ["10v", "v10"],
            "msl": ["msl", "mslp"],
        }

        surface_list = []
        for var_name in self.surface_vars:
            da = self._get_var(ds, surface_aliases.get(var_name, [var_name]))

            arr = da.isel(time=time_idx).values.astype(np.float32)
            arr = np.asarray(arr)
            arr = np.squeeze(arr)

            # HRES latitude: -90 -> 90
            # ERA5/model expected latitude: 90 -> -90
            arr = arr[..., ::-1, :].copy()

            surface_list.append(torch.from_numpy(arr))

        input_upper = torch.stack(upper_list, dim=0)      # [5, 13, 721, 1440]
        input_surface = torch.stack(surface_list, dim=0)  # [4, 721, 1440]

        return input_upper, input_surface

    def normlize(self, upper, surface):
        upper = (upper - self.mean_upper) / self.std_upper
        surface = (surface - self.mean_surface) / self.std_surface
        return upper, surface

    def __len__(self):
        return max(0, (self.idx_len - self.steps - 1) // 2)

    def __getitem__(self, idx):
        file_idx = idx * 2 + 1

        # HRES input
        # idx=0 -> HRES time[1], time[2]
        input_upper, input_surface = self._read_hres_upper_surface(file_idx)
        input_upper1, input_surface1 = self._read_hres_upper_surface(file_idx + 1)

        input_upper, input_surface = self.normlize(input_upper, input_surface)
        input_upper1, input_surface1 = self.normlize(input_upper1, input_surface1)

        input_upper = torch.cat([input_upper, input_upper1], dim=0)        # [10, 13, 721, 1440]
        input_surface = torch.cat([input_surface, input_surface1], dim=0)  # [8, 721, 1440]

        # ERA5 target
        target_uppers = []
        target_surfaces = []

        for step in range(1, self.steps + 1):
            target_idx = self.dir_files[file_idx + 1 + step]
            target_dict = torch.load(
                f"{self.era5_root}/{self.file_pre}/{target_idx}",
                weights_only=False
            )

            target_uppers.append(target_dict["input_upper"])
            target_surfaces.append(target_dict["input_surface"])

        target_upper = torch.stack(target_uppers, dim=0)      # [T, 5, 13, 721, 1440]
        target_surface = torch.stack(target_surfaces, dim=0)  # [T, 4, 721, 1440]

        return input_upper, input_surface, target_upper, target_surface
    
