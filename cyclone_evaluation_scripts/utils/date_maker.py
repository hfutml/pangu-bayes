# %%
import cdsapi
import numpy as np
import pickle
import subprocess
import xarray as xr
import pandas as pd
import os

# %% Load the original ibtracs data
ibtracs_path = os.path.join(
    os.path.join(os.curdir, "data", "ibtracs"),
    "ibtracs.ALL.list.v04r01.csv",
)
df = pd.read_csv(ibtracs_path, dtype=str, skiprows=[1], na_filter=False)
# %% parse the datetimes
df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"])

# Select the years after 1980
df = df[df["ISO_TIME"].dt.year >= 1980]
# %%

# and the "standard" timesteps
hours_to_select = [
    0,
    6,
    12,
    18,
]
df = df[df["ISO_TIME"].dt.hour.isin(hours_to_select)]

# Adding the "negative" lead times to be able to handle genesis
max_lead = 24
step = 6
iso_times = df["ISO_TIME"].unique()
# set up a timedelta64 array for up to negative max_lead time with step hours
timedeltas = np.arange(-np.timedelta64(max_lead, "h"), 0, np.timedelta64(step, "h"))
# create a new array with the original iso_times and the negative lead times
iso_copy = iso_times.copy()
for delta in timedeltas:
    iso_copy = np.hstack([iso_copy, iso_times + delta])
# convert to datetime index
iso_times = pd.to_datetime(iso_copy).unique()

years = iso_times.year.unique()

# overwrite with 2023
years = [2023]

era5_dler_script = os.path.join(os.curdir, "utils", "ERA5_dler.py")
# %%
for year in years:
    temp_df = df[df["ISO_TIME"].dt.year == year]
    months = temp_df["ISO_TIME"].dt.month.unique()
    for month in months:
        if month > 9:
            continue
        days = temp_df["ISO_TIME"][
            temp_df["ISO_TIME"].dt.month == month
        ].dt.day.unique()

        for source in ["pressure", "surface"]:

            if source == "pressure":
                datavars = [
                    "geopotential",
                    "relative_humidity",
                    "temperature",
                    "u_component_of_wind",
                    "v_component_of_wind",
                    "vertical_velocity",
                    "specific_humidity",
                ]
            elif source == "surface":
                datavars = [
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "100m_u_component_of_wind",
                    "100m_v_component_of_wind",
                    "2m_temperature",
                    "Geopotential",
                    "land_sea_mask",
                    "total_precipitation",
                    "mean_sea_level_pressure",
                    "surface_pressure",
                    "total_column_water_vapour",
                ]

            # create the download path
            download_path = os.path.join(
                "<ANONYMIZED>/ERA5_data",
                f"ERA5_{year}_{str(month).zfill(2)}_{source}.nc",
            )
            if not os.path.exists(download_path):
                os.makedirs(download_path)
            # run the era5_dler script
            subprocess.run(
                [
                    "python",
                    era5_dler_script,
                    "--year",
                    str(year),
                    "--month",
                    str(month),
                    "--days",
                    str(list(days)),
                    "--source",
                    source,
                    "--download_path",
                    download_path,
                    "--datavars",
                    str(datavars).replace("\n", ""),
                ]
            )
