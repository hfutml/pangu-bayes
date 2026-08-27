#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 11:44:20 2022

Python script used to download ERA5 data from the CDS API into single files
with multiple pressure levels.
"""
# %%
from unittest import case
import cdsapi
import os
import argparse
import json

# %%
parser = argparse.ArgumentParser(description="Download ERA5 data")

# read in the bash arguments
parser.add_argument(
    "--datavars", type=str, default="[divergence]", help="variables to download"
)
parser.add_argument("--source", type=str, default="surface", help="source of the data")
parser.add_argument("--year", type=int, default=2023, help="year to download")
parser.add_argument("--month", type=int, default=1, help="Month to download")
parser.add_argument("--days", type=str, default="[1]", help="days to download")
parser.add_argument(
    "--download_path", type=str, default=".", help="path to download the data"
)

args = parser.parse_args()
# %%
# load client interface
client = cdsapi.Client(timeout=86000, quiet=False, sleep_max=900, retry_max=1000)

# load the variables to download
# datavars = json.loads(args.datavars)
datavars = args.datavars.replace("[", "").replace("]", "").replace("'", "").split(",")
datavars = [var.strip() for var in datavars]

if args.source == "pressure":
    # Pressure levels to download according to NeWM requirements
    full_pressure = [
        "1000",
        "925",
        "850",
        "700",
        "600",
        "500",
        "400",
        "300",
        "250",
        "200",
        "150",
        "100",
        "50",
    ]


# Download every 6 hours, which is the maximum (interpolated) resolution for ibtracs
times = [
    "00:00",
    # "03:00",
    "06:00",
    # "09:00",
    "12:00",
    # "15:00",
    "18:00",
    # "21:00",
]

years = [
    str(args.year),
]

months = [
    f"{args.month:02d}",
]

# if statement to get the list of days according to the month - ignore arguments
if args.month in [1, 3, 5, 7, 8, 10, 12]:
    days = [f"{day:02d}" for day in range(1, 32)]
elif args.month in [4, 6, 9, 11]:
    days = [f"{day:02d}" for day in range(1, 31)]
elif args.month == 2:
    if args.year % 4 == 0 and (args.year % 100 != 0 or args.year % 400 == 0):
        days = [f"{day:02d}" for day in range(1, 30)]  # leap year
    else:
        days = [f"{day:02d}" for day in range(1, 29)]
else:
    raise ValueError("Invalid month provided. Please provide a valid month (1-12).")

# days = json.loads(args.days)

directory_path = args.download_path
if not directory_path.endswith("/"):
    directory_path += "/"
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

filename = f"ERA5_{args.year}-{args.month:02d}_{args.source}_{datavars[0]}.nc"
target_path = os.path.join(directory_path, filename)
if os.path.exists(target_path):
    print(f"File {filename} already exists")
    exit()


data_params = {
    "product_type": ["reanalysis"],
    "data_format": "netcdf",  # "grib"
    "variable": datavars,
    "year": years,
    "month": months,
    "day": [f"{int(day):02d}" for day in days],
    "time": times,
    "download_format": "unarchived",
}

# set the era5 data origin
if args.source == "surface":
    data_origin = "reanalysis-era5-single-levels"
elif args.source == "pressure":
    data_origin = "reanalysis-era5-pressure-levels"
    data_params["pressure_level"] = full_pressure
# %%
# print(data_origin, data_params, target_path)
client.retrieve(name=data_origin, request=data_params, target=target_path)

# %%
