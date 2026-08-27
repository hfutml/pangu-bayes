import os
import xarray as xr
import numpy as np
import torch
import pandas as pd
import datetime
import pickle as pkl


def save2nc(
    input_upper_list,
    input_surface_list,
    time_start="2023-01-01T00:00",
    sum_nums=20,
    noise_idx=0,
    global_rank=0,
):
    """
    :param input_upper_list: lead_times 's upper variables list
    :param input_surface_list: lead_times 's surface variables list
    """

    surface_index = {"t2m": 0, "u10": 1, "v10": 2, "msl": 3}
    upper_index = {"z": 0, "q": 1, "t": 2, "u": 3, "v": 4}
    upper_level = {
        50: 0,
        100: 1,
        150: 2,
        200: 3,
        250: 4,
        300: 5,
        400: 6,
        500: 7,
        600: 8,
        700: 9,
        850: 10,
        925: 11,
        1000: 12,
    }

    unit_dict = {
        "t2m": "K",
        "10u": "m/s",
        "10v": "m/s",
        "msl": "hPa",
        "z": "m",
        "q": "kg/kg",
        "t": "K",
        "u": "m/s",
        "v": "m/s",
    }
    title_dict = {
        "t2m": "2m temperature",
        "10u": "10m u component of wind",
        "10v": "10m v component of wind",
        "msl": "Mean Sea Level Pressure",
        "z": "Geopotential",
        "q": "Specific humidity",
        "t": "Temperature",
        "u": "U component of wind",
        "v": "V component of wind",
    }

    # mainly for aligning the longitude and latitude of the nc file
    nc_file = "./dataset/t2m_1979_0.nc"

    var_info_list = [
        {
            "var": "msl",
            "type": "surface",
            "level": None,
            "title": title_dict["msl"],
            "unit": unit_dict["msl"],
        },
        {
            "var": "u10",
            "type": "surface",
            "level": None,
            "title": title_dict["10u"],
            "unit": unit_dict["10u"],
        },
        {
            "var": "v10",
            "type": "surface",
            "level": None,
            "title": title_dict["10v"],
            "unit": unit_dict["10v"],
        },
        {
            "var": "z",
            "type": "upper",
            "level": 300,
            "title": title_dict["z"] + str(300),
            "unit": unit_dict["z"],
        },
        {
            "var": "z",
            "type": "upper",
            "level": 500,
            "title": title_dict["z"] + str(500),
            "unit": unit_dict["z"],
        },
    ]

    ds = xr.open_dataset(nc_file)
    lons = ds["longitude"].values
    lats = ds["latitude"].values
    time = pd.date_range(time_start, periods=sum_nums + 1, freq="6h")
    time = time[1:]

    save_val_list = ["msl", "u10", "v10", "z300", "z500"]

    combined_data = {var: None for var in save_val_list}

    for var_info in var_info_list:
        for input_upper_init, input_surface_init in zip(
            input_upper_list, input_surface_list
        ):
            select_var = var_info["var"]
            select_type = var_info["type"]
            select_level = var_info["level"]
            select_index = (
                surface_index[select_var]
                if select_type == "surface"
                else upper_index[select_var]
            )
            if select_type == "upper":
                select_var = select_var + str(select_level)
            if select_type == "surface":
                if input_surface_init.dim() == 4:
                    input_surface_init = input_surface_init.view(4, 721, 1440)
                input_init = input_surface_init[select_index, :, :]  # initial field
            elif select_type == "upper":
                select_level = upper_level[select_level]
                if input_upper_init.dim() == 5:
                    input_upper_init = input_upper_init.view(5, 13, 721, 1440)
                input_init = input_upper_init[select_index, select_level, :, :]

            data_init = input_init.cpu().numpy()

            if combined_data[select_var]:
                combined_data[select_var].append(data_init)
            else:
                combined_data[select_var] = [data_init]
    ds = xr.Dataset(
        data_vars={
            select_var: (["time", "lat", "lon"], np.array(combined_data[select_var]))
            for select_var in combined_data
        },
        coords={
            "time": time,
            "lat": lats,
            "lon": lons,
        },
    )

    ds.attrs["title"] = "Pangu weather prediction output"

    ds.attrs = {"longname": title_dict["msl"], "units": unit_dict["msl"]}

    dir = "./result/"

    output_path = f"{dir}Pangu_{time_start}_{global_rank}_{noise_idx}.nc"
    ds.to_netcdf(
        output_path,
        encoding={
            var: {"zlib": True, "complevel": 4, "dtype": "float64"}
            for var in ds.data_vars
        },
    )


def extract_patch_numpy(field, lat, lon, mask, r=120, fill1=True):
    """
    Extract a patch from a field of shape (721, 1440) centered at a given latitude and longitude.
    The patch has a shape of (241, 241) and apply mask.

    field: np.ndarray, shape of (721, 1440)
    lat: float, target latitude
    lon: float, target longitude
    mask: np.ndarray, shape of (241, 241)
    r: int, radius (default 120, corresponding to a window size of 241)
    """
    if fill1:
        mask = torch.tensor(mask)
        mask = (mask > 0).float()
        mask = torch.where(mask > 0, torch.tensor(1), torch.tensor(0))
        mask = mask.cpu().numpy()

    H, W = field.shape  # 721, 1440

    # Coordinate to Index
    # Latitude: 90 -> 0, -90 -> 720. Formula: (90 - lat) / 0.25
    center_row = int(round((90 - lat) / 0.25))

    # Longitude: -180~180 transfer to 0~360, then divide by 0.25
    if lon < 0:
        lon = lon + 360
    center_col = (
        int(round(lon / 0.25)) % W
    )  # % W prevent exactly 360 becoming out of range

    # generate longitude indices (handle circular boundary)
    col_indices = np.arange(center_col - r, center_col + r + 1) % W

    # generate latitude indices (handle circular boundary)
    row_start = center_row - r
    row_end = center_row + r + 1

    # initialize patch with zeros (or np.nan)
    full_patch = np.zeros((2 * r + 1, 2 * r + 1))

    valid_row_start = max(0, row_start)
    valid_row_end = min(H, row_end)

    if valid_row_start >= valid_row_end:
        return full_patch * mask

    # extract data from field
    temp_strip = field[:, col_indices]
    paste_row_start = valid_row_start - row_start
    paste_row_end = paste_row_start + (valid_row_end - valid_row_start)

    full_patch[paste_row_start:paste_row_end, :] = temp_strip[
        valid_row_start:valid_row_end, :
    ]

    result = full_patch * mask

    return full_patch, result


def save2pt(full_ens_upper, full_ens_surface, now_time, lon_list, lat_list, sid_list):
    """
    full_ens_upper and full_ens_surface to pt file
    """

    now_time = now_time[0]

    data_dict = pkl.load(open("mask_dict.pkl", "rb"))
    steps, C, H, W = full_ens_upper.shape
    for sid, lon, lat in zip(sid_list, lon_list, lat_list):
        sid, lon, lat = sid[0], lon[0].cpu().numpy(), lat[0].cpu().numpy()
        ens_mask_list = []
        for i in range(steps):
            time_steps = (i + 1) * 6
            mask = data_dict[time_steps]
            mask_list = []
            mask_list = []
            for j in range(3):
                ens_surface = full_ens_surface[i, j, :, :].cpu().numpy()
                _, ens_surface_mask = extract_patch_numpy(ens_surface, lat, lon, mask)
                mask_list.append(torch.from_numpy(ens_surface_mask))
            for j in range(2):
                ens_upper = full_ens_upper[i, j, :, :].cpu().numpy()
                _, ens_upper_mask = extract_patch_numpy(ens_upper, lat, lon, mask)
                mask_list.append(torch.from_numpy(ens_upper_mask))
            mask_tensor = torch.stack(mask_list, dim=0)
            ens_mask_list.append(mask_tensor)
        ens_mask_tensor = torch.stack(ens_mask_list, dim=0)
        save_pt_dict = {
            "u10": ens_mask_tensor[:, 0, :, :],
            "v10": ens_mask_tensor[:, 1, :, :],
            "msl": ens_mask_tensor[:, 2, :, :],
            "z500": ens_mask_tensor[:, 3, :, :],
            "t850": ens_mask_tensor[:, 4, :, :],
        }

        dir = "Anonymous_path/"
        os.makedirs(dir, exist_ok=True)

        date_object = datetime.datetime.strptime(now_time, "%Y-%m-%dT%H:%M")
        month = date_object.month
        year = date_object.year

        save_dir = f"{dir}{year}-{month:02d}/"
        os.makedirs(save_dir, exist_ok=True)

        output_path = save_dir + f"Pangu_{sid}.pt"
        torch.save(save_pt_dict, output_path)
