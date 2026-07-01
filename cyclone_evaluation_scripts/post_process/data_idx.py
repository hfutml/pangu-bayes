import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import netCDF4 as nc
import pandas as pd
import json
from datetime import datetime, timedelta


class WeatherDataset40inference(Dataset):
    def __init__(self, patch_size, steps, filter_time_range_file=None):
        super().__init__()
        self.patch_size = patch_size
        self.steps = steps

        self.file_pre = "test"
        self.path_dir = "./dataset/"
        df = pd.read_csv(self.path_dir + "2023_time_file-index_dict.csv")
        df_train = pd.read_csv(self.path_dir + "2013-2022_time_file-index_dict.csv")
        self.index_dict_total = dict(zip(df["Timestamp"], df["Index"]))
        self.index_dict_total_train = dict(
            zip(df_train["Timestamp"], df_train["Index"])
        )
        # print(self.index_dict_total)

        self.data_path_dir = "./dataset/" + self.file_pre + "/"

        if filter_time_range_file is None:
            raise ValueError(
                "filter_time_range_file 未提供：请指定时间范围过滤的 json 文件路径"
            )
        with open(filter_time_range_file, "r") as f:
            data_dict = json.load(f)
        self.train_sets = data_dict
        self.train_times = sorted(
            [
                k
                for k in data_dict
                if datetime.strptime(k, "%Y-%m-%dT%H:%M").minute == 0
                and datetime.strptime(k, "%Y-%m-%dT%H:%M").hour in [0, 6, 12, 18]
            ]
        )
        excluded_times = sorted([k for k in data_dict if k not in self.train_times])

        print("The time steps that do not meet the requirements are:\n", excluded_times)

        self.idx_len = int(len(self.train_times))
        # self.idx_len = 1

        self.mean_surface = (
            torch.load('../ERA5_examples/statistic/surface_mean_40.pt', weights_only=False)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        self.mean_upper = (
            torch.load('../ERA5_examples/statistic/upper_mean_40.pt', weights_only=False)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        self.std_surface = (
            torch.load('../ERA5_examples/statistic/surface_std_40.pt', weights_only=False)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        self.std_upper = (
            torch.load('../ERA5_examples/statistic/upper_std_40.pt', weights_only=False)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

    def normlize(self, upper, surface):
        upper = (upper - self.mean_upper) / self.std_upper
        surface = (surface - self.mean_surface) / self.std_surface
        return upper, surface

    def __len__(self):
        return self.idx_len

    def shift_time_by_hours(self, time_str, hours):
        # time_str = "2025-01-01T00:00"
        dt = datetime.fromisoformat(time_str)
        new_dt = dt + timedelta(hours=hours)
        new_time_str = new_dt.isoformat()
        year = new_dt.year
        year_str = str(year)
        return new_time_str[:-3], year_str

    def __getitem__(self, idx):
        tim = self.train_times[idx]
        now_time = tim
        lon_list, lat_list, sid_list = [], [], []
        for item in self.train_sets[now_time]:
            lon_list.append(item["lon"])
            lat_list.append(item["lat"])
            sid_list.append(item["id"])
        pre_time, year_str = self.shift_time_by_hours(now_time, -6)

        # idx2 = self.dir_files[file_idx + 1 + self.steps]

        if year_str == "2022":
            data_path_dir = self.data_path_dir
            input_dict = torch.load(
                f"{data_path_dir}{self.index_dict_total_train[pre_time]}.pt"
            )
        elif year_str <= "2021":
            data_path_dir = self.data_path_dir.replace("test", "train")

        else:
            data_path_dir = self.data_path_dir

            input_dict = torch.load(
                f"{data_path_dir}{self.index_dict_total[pre_time]}.pt"
            )

        input_upper0, input_surface0 = (
            input_dict["input_upper"],
            input_dict["input_surface"],
        )

        (
            input_upper_nrom0,
            input_surface_nrom0,
        ) = self.normlize(input_upper0, input_surface0)

        year_str = str(now_time)[:4]

        if year_str == "2022":
            data_path_dir = self.data_path_dir
            input_dict = torch.load(
                f"{data_path_dir}{self.index_dict_total_train[now_time]}.pt"
            )
        elif year_str <= "2021":
            data_path_dir = self.data_path_dir.replace("test", "train")
            input_dict = torch.load(
                f"{data_path_dir}{self.index_dict_total_train[now_time]}.pt"
            )
        else:
            data_path_dir = self.data_path_dir
            input_dict = torch.load(
                f"{data_path_dir}{self.index_dict_total[now_time]}.pt"
            )
        input_upper1, input_surface1 = (
            input_dict["input_upper"],
            input_dict["input_surface"],
        )

        input_upper_nrom1, input_surface_nrom1 = self.normlize(
            input_upper1, input_surface1
        )

        input_upper, input_surface = (
            torch.cat([input_upper_nrom0, input_upper_nrom1], dim=0),
            torch.cat([input_surface_nrom0, input_surface_nrom1], dim=0),
        )

        return input_upper, input_surface, now_time, lon_list, lat_list, sid_list
