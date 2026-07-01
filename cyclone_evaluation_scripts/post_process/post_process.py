# %%
import json
import torch
import pickle as pkl
from tqdm import tqdm
from datetime import datetime
import warnings
import pandas as pd
import pandas as pd

import os
import pandas as pd

import os
import pandas as pd

# ================= 配置区域 =================
INPUT_FILE = "../ibtracs/ibtracs_2013_2023.csv"
OUTPUT_DIR = "./dataset/"

TRAIN_YEARS = range(2013, 2023)
TEST_YEARS = [2023]

ALL_HOURS = [0, 6, 12, 18]
KEPT_HOURS = [0, 12]
LEAD_TIMES = list(range(6, 121, 6))


def prepare_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Reading: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    df = df[df["ISO_TIME"].dt.hour.isin(ALL_HOURS)].copy()

    for col in ["USA_WIND", "USA_PRES"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)

    for hours in LEAD_TIMES:
        steps = hours // 6

        future_wind = df.groupby("SID")["USA_WIND"].shift(-steps)
        future_pres = df.groupby("SID")["USA_PRES"].shift(-steps)

        df[f"Target_Delta_Wind_{hours}h"] = future_wind - df["USA_WIND"]
        df[f"Target_Delta_Pres_{hours}h"] = future_pres - df["USA_PRES"]

    df = df[
        df["Target_Delta_Wind_6h"].notna()
        & df["Target_Delta_Pres_6h"].notna()
    ].copy()

    if "NATURE" in df.columns:
        df["NATURE"] = df["NATURE"].astype(str).str.strip()
        df = df[df["NATURE"].isin(["TS", "SS"])].copy()

    df = df[df["ISO_TIME"].dt.hour.isin(KEPT_HOURS)].copy()
    df["Year"] = df["ISO_TIME"].dt.year

    train_df = df[df["Year"].isin(TRAIN_YEARS)].copy()
    test_df = df[df["Year"].isin(TEST_YEARS)].copy()

    full_target_cols = []

    for hours in LEAD_TIMES:
        full_target_cols.append(f"Target_Delta_Wind_{hours}h")
        full_target_cols.append(f"Target_Delta_Pres_{hours}h")

    before_train = len(train_df)

    train_df = train_df.dropna(
        subset=full_target_cols,
        how="any"
    ).copy()

    after_train = len(train_df)

    train_path = os.path.join(OUTPUT_DIR, "2013_2022_00_12z.csv")
    test_path = os.path.join(OUTPUT_DIR, "2023_00_12z.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("-" * 40)
    print(f"Train samples before 120h filtering: {before_train}")
    print(f"Train samples after 120h filtering:  {after_train}")
    print(f"Removed train samples:               {before_train - after_train}")
    print(f"Test samples:                         {len(test_df)}")
    print(f"Train file:                           {train_path}")
    print(f"Test file:                            {test_path}")
    print("-" * 40)


prepare_data()











# df = pd.read_csv("./dataset/test/2023_00_12z.csv")
# id_vars = [col for col in df.columns if not col.startswith("Target_Delta_")]

# df_long = pd.wide_to_long(
#     df,
#     stubnames=["Target_Delta_Wind", "Target_Delta_Pres"],
#     i=id_vars,
#     j="Forecast_Hour",
#     sep="_",
#     suffix=r"\d+h",
# )

# df_long = df_long.reset_index()
# df_long["Forecast_Hour"] = df_long["Forecast_Hour"].str.replace("h", "").astype(int)
# df_long["Target_Delta_Wind"] = pd.to_numeric(df_long["Target_Delta_Wind"], errors="coerce")
# df_long["Target_Delta_Pres"] = pd.to_numeric(df_long["Target_Delta_Pres"], errors="coerce")
# df_long = df_long.dropna(subset=["Target_Delta_Wind", "Target_Delta_Pres"], how="any")

# print(df_long.head())

# df_long.to_csv("./dataset/test/test_2023_00_12z_reshaped_data.csv", index=False)














# warnings.filterwarnings("ignore")
# save_dir = "ANONYMOUS_PATH"
# with open("./dataset/test_set.json", "r") as f:
#     data_dict = json.load(f)
# df = pd.read_csv("./dataset/test/test_2023_00_12z_reshaped_data.csv")
# df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"])
# df["key"] = (
#     df["SID"].astype(str)
#     + "_"
#     + df["ISO_TIME"].astype(str)
#     + "_"
#     + df["Forecast_Hour"].astype(str)
# )

# statistical_variables = [
#     "V10_max",
#     "P0_min",
#     "V10_range",
#     "P0_range",
#     "z500_min",
#     "t850_range",
# ]
# for var in statistical_variables:
#     df[var] = None

# base_dir = save_dir

# tc_list = list(data_dict.keys())
# for time_step in tqdm(tc_list):
#     for tc in data_dict[time_step]:
#         tc_id = tc["track_id"]
#         tc_tim = datetime.strptime(time_step, "%Y-%m-%dT%H:%M")
#         tc_ios_time = tc_tim.strftime("%Y-%m-%d %H:%M:%S")
#         file_name = "Pangu_" + tc["id"] + ".pt"
#         month = tc_tim.month
#         year = tc_tim.year
#         file_path = f"{base_dir}/{year}-{month:02d}/{file_name}"
#         prediction = torch.load(file_path)

#         v10_tensor = prediction["v10"]
#         u10_tensor = prediction["u10"]
#         msl_tensor = prediction["msl"]
#         z500_tensor = prediction["z500"]
#         t850_tensor = prediction["t850"]
#         steps, _, _ = v10_tensor.shape
#         mask_data_dict = pkl.load(open("mask_dict.pkl", "rb"))

#         for step in range(steps):
#             v10 = v10_tensor[step]
#             u10 = u10_tensor[step]
#             msl = msl_tensor[step]
#             z500 = z500_tensor[step]
#             t850 = t850_tensor[step]
#             mask_linear = mask_data_dict[(step + 1) * 6]
#             valid_mask = mask_linear > 0

#             V10m = torch.sqrt(v10**2 + u10**2)
#             V10m = V10m[valid_mask]
#             msl = msl[valid_mask]
#             z500 = z500[valid_mask]
#             t850 = t850[valid_mask]

#             V10m_max = torch.max(V10m)
#             P0_min = torch.min(msl)
#             P0_range = torch.max(msl) - P0_min
#             V10m_range = V10m_max - torch.min(V10m)
#             z500_min = torch.min(z500)
#             t850_range = torch.max(t850) - torch.min(t850)

#             SV_dict = {
#                 "V10_max": V10m_max.numpy(),
#                 "P0_min": P0_min.numpy(),
#                 "V10_range": V10m_range.numpy(),
#                 "P0_range": P0_range.numpy(),
#                 "z500_min": z500_min.numpy(),
#                 "t850_range": t850_range.numpy(),
#             }
#             for var in statistical_variables:
#                 df.loc[df["key"] == f"{tc_id}_{tc_ios_time}_{(step + 1) * 6}", var] = (
#                     SV_dict[var]
#                 )


# df.to_csv("statistical_variables_test1.csv", index=False)
