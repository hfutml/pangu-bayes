# %%
import xarray as xr
import pandas as pd
import numpy as np
import os
import huracanpy
import argparse


parser = argparse.ArgumentParser(description="Track matching Script")
parser.add_argument("--initial_dir", type=str, default="./", help="inference result root directory")
parser.add_argument("--ibtracs_path",type=str,default="./data/ibtracs/ibtracs.ALL.list.v04r01.csv",help="ibtracs dataset path")
parser.add_argument("--method", type=str, default="SPPT", help="experiment method")
parser.add_argument("--model", type=str, default="pangu", help="model name")
parser.add_argument("--save_dir", type=str, default="./", help="save directory")
args = parser.parse_args()

initial_dir = args.initial_dir
method = args.method
model = args.model

# Get the list of files in the directory
file_dir = initial_dir + f"{method}/tracks/"

# Load ibtracs
ibtracs_path = args.ibtracs_path
ibtracs = pd.read_csv(ibtracs_path, keep_default_na=False)

# remove the first row
ibtracs = ibtracs.iloc[1:]

# Keep only the SID, ATCF_ID, lat, lon, ISO_TIME, USA_WIND and USA_PRESSURE columns
ibtracs = ibtracs[
    ["SID", "USA_ATCF_ID", "LAT", "LON", "ISO_TIME", "SEASON", "USA_WIND", "USA_PRES"]
]
# filter out rows with non-numeric values in the SEASON column
# conver lat, lon to float
ibtracs["LAT"] = ibtracs["LAT"].astype(float)
ibtracs["LON"] = ibtracs["LON"].astype(float)

# if the longtitude is less than 0, add 360
ibtracs.loc[ibtracs["LON"] < 0, "LON"] += 360

ibtracs = ibtracs[pd.to_numeric(ibtracs["SEASON"], errors="coerce").notnull()]
# turn season to int
ibtracs["SEASON"] = ibtracs["SEASON"].astype(int)
ibtracs = ibtracs[ibtracs["SEASON"] > 2020]
# turn ISO_TIME to datetime
ibtracs["ISO_TIME"] = pd.to_datetime(ibtracs["ISO_TIME"])
# select year 2023
ibtracs = ibtracs[ibtracs["ISO_TIME"].dt.year == 2023]

# Create an empty DataFrame with the desired columns
columns = [
    "SID",
    "Initial Time",
    "Valid Time",
    "ensemble_idx",
    "wind max",
    "pressure min",
    "lat",
    "lon",
]
df = pd.DataFrame(columns=columns)
ib = None
file_list = os.listdir(file_dir)
file_extract_time = [file.split("_")[1] for file in file_list]
file_extract_gpu_idx = [int(file.split("_")[2]) for file in file_list]
file_extract_noise_idx = [int(file.split("_")[3].split(".")[0]) for file in file_list]

for file, extract_time, gpu_idx, noise_idx in zip(
    file_list, file_extract_time, file_extract_gpu_idx, file_extract_noise_idx
):
    ensemble_idx = gpu_idx * 6 + noise_idx
    init_date = extract_time
    if model == "aifs":
        init_date = file.split("_")[1][5:]
    else:
        init_date = file.split("_")[1]
    init_datetime = pd.to_datetime(init_date)

    try:
        detected = huracanpy.load(os.path.join(file_dir, file))
    except Exception as e:
        print(f"Error loading detected data: {e}")
        continue

    # detected is a netcdf file; flip the lat values if model is "aifs"
    if model == "aifs":
        # multiply lat values by -1
        detected["lat"] = detected["lat"] * -1

    if ib is None:
        try:
            ib = huracanpy._data.load(
                filename=ibtracs, source="csv", load_function=lambda x: x
            )
        except Exception as e:
            print(f"Error loading ibtracs data: {e}")
            raise e
        ib = ib.rename({"sid": "track_id"})

    match = huracanpy.assess.match([ib, detected], ["ibtracs", "detected"])

    # iterate through the match df rows
    for index, row in match.iterrows():
        # Get the values for the current row
        sid = row["id_ibtracs"]
        initial_time = init_datetime
        temp_track = detected.where(detected.track_id == row["id_detected"], drop=True)
        detected_wind = temp_track.wind10.values
        detected_pressure = temp_track.slp.values
        detected_lat = temp_track.lat.values
        # switch lat sign for all values
        # detected_lat = np.where(detected_lat > 0, detected_lat, -detected_lat)
        detected_lon = temp_track.lon.values
        detected_time = temp_track.time.values

        # Create a new DataFrame with the current values
        temp_df = pd.DataFrame(
            {
                "SID": sid,
                "Initial Time": initial_time,
                "Valid Time": detected_time,
                "ensemble_idx": ensemble_idx,
                "wind max": detected_wind,
                "pressure min": detected_pressure,
                "lat": detected_lat,
                "lon": detected_lon,
            }
        )

        if df.empty:
            df = temp_df
        else:
            # Concatenate the new DataFrame with the existing one
            df = pd.concat([df, temp_df], ignore_index=True)

# convert wind to knots
df["wind max"] = df["wind max"] * 1.94384
# convert pressure to hPa
df["pressure min"] = df["pressure min"] / 100

df = df.sort_values(by=["SID", "Initial Time", "Valid Time", "ensemble_idx"])
# Save the DataFrame to a CSV file
df.to_csv(
    f"{args.save_dir}/{method}/match_result_{method}.csv",
    index=False,
)
