import numpy as np
import pandas as pd
from ri_metric import calc_csi_pss_from_binary

def build_strict_24h_member_level_ri_records(
    df,
    id_cols=("SID", "ISO_TIME"),
    forecast_hour_col="Forecast_Hour",
    true_future_wind_col="true_future_wind",
    pred_mu_delta_col="pred_mu_Delta_Wind",
    pred_sigma_delta_col="pred_sigma_Delta_Wind",
    init_wind_col="USA_WIND",
    ri_threshold=30.0,
    n_samples=50,
    random_seed=42,
):

    rng = np.random.default_rng(random_seed)
    df = df.copy().reset_index(drop=True)

    # 基础检查
    needed = [forecast_hour_col, true_future_wind_col, pred_mu_delta_col, pred_sigma_delta_col, init_wind_col]
    for c in list(id_cols) + needed:
        if c not in df.columns:
            raise ValueError(f"缺少必要列: {c}")

    # 采样未来风速
    mu_delta = df[pred_mu_delta_col].values
    sigma_delta = np.clip(df[pred_sigma_delta_col].values, 1e-6, None)
    init_wind = df[init_wind_col].values

    sampled_delta = rng.normal(
        loc=mu_delta[:, None],
        scale=sigma_delta[:, None],
        size=(len(df), n_samples)
    )  # [N, M]

    # 每个成员的未来风速 V(t+fh)
    sampled_future_wind = init_wind[:, None] + sampled_delta

    member_records = []
    for m in range(n_samples):
        sub = df.copy()
        sub["ensemble_member"] = m + 1
        sub["sampled_future_wind"] = sampled_future_wind[:, m]
        member_records.append(sub)

    member_level_df = pd.concat(member_records, axis=0, ignore_index=True)

    # 构造一个 group key：同一气旋、同一起报时间
    group_cols = list(id_cols) + ["ensemble_member"]
    base_group_cols = list(id_cols)

    # 先按真实值计算 24h 差分
    df_true = df.copy().sort_values(list(id_cols) + [forecast_hour_col]).reset_index(drop=True)

    # 为了能用 fh-24 对齐，建立 key
    df_true["prev_fh"] = df_true[forecast_hour_col] - 24

    prev_true = df_true[list(id_cols) + [forecast_hour_col, true_future_wind_col]].copy()
    prev_true = prev_true.rename(columns={
        forecast_hour_col: "prev_fh",
        true_future_wind_col: "true_future_wind_prev24"
    })

    df_true = df_true.merge(prev_true, on=list(id_cols) + ["prev_fh"], how="left")
    df_true["true_24h_delta_wind"] = df_true[true_future_wind_col] - df_true["true_future_wind_prev24"]

    # fh == 24 时，前一时刻相当于初始风速
    mask_24 = df_true[forecast_hour_col] == 24
    df_true.loc[mask_24, "true_24h_delta_wind"] = (
        df_true.loc[mask_24, true_future_wind_col] - df_true.loc[mask_24, init_wind_col]
    )

    # 真实 RI
    df_true["obs_RI"] = (df_true["true_24h_delta_wind"] >= ri_threshold).astype(int)

    # 把真实 24h 信息并回成员表
    merge_true_cols = list(id_cols) + [forecast_hour_col, "true_24h_delta_wind", "obs_RI"]
    member_level_df = member_level_df.merge(
        df_true[merge_true_cols],
        on=list(id_cols) + [forecast_hour_col],
        how="left"
    )

    # 对每个成员计算预测 24h 差分
    member_level_df = member_level_df.sort_values(group_cols[:-1] + ["ensemble_member", forecast_hour_col]).reset_index(drop=True)
    member_level_df["prev_fh"] = member_level_df[forecast_hour_col] - 24

    prev_pred = member_level_df[group_cols + [forecast_hour_col, "sampled_future_wind"]].copy()
    prev_pred = prev_pred.rename(columns={
        forecast_hour_col: "prev_fh",
        "sampled_future_wind": "sampled_future_wind_prev24"
    })

    member_level_df = member_level_df.merge(
        prev_pred,
        on=group_cols + ["prev_fh"],
        how="left"
    )

    member_level_df["pred_24h_delta_wind"] = (
        member_level_df["sampled_future_wind"] - member_level_df["sampled_future_wind_prev24"]
    )

    # fh == 24 时，用未来风速 - 初始风速
    mask_24_member = member_level_df[forecast_hour_col] == 24
    member_level_df.loc[mask_24_member, "pred_24h_delta_wind"] = (
        member_level_df.loc[mask_24_member, "sampled_future_wind"] -
        member_level_df.loc[mask_24_member, init_wind_col]
    )

    # fh < 24 无法定义严格24h增强，删掉
    member_level_df = member_level_df[member_level_df[forecast_hour_col] >= 24].copy()

    # 预测 RI
    member_level_df["pred_RI"] = (
        member_level_df["pred_24h_delta_wind"] >= ri_threshold
    ).astype(int)

    return member_level_df


def summarize_strict_24h_ri_by_lead_time(
    member_level_df,
    forecast_hour_col="Forecast_Hour",
):
    rows = []
    for fh, sub in member_level_df.groupby(forecast_hour_col):
        metric = calc_csi_pss_from_binary(sub["obs_RI"].values, sub["pred_RI"].values)
        rows.append({
            "Forecast_Hour": fh,
            "N_member_records": len(sub),
            "N_obs_RI": int(sub["obs_RI"].sum()),
            "N_pred_RI": int(sub["pred_RI"].sum()),
            "Obs_RI_Rate": float(sub["obs_RI"].mean()),
            "Pred_RI_Rate": float(sub["pred_RI"].mean()),
            **metric,
        })
    return pd.DataFrame(rows).sort_values("Forecast_Hour").reset_index(drop=True)



test_result_df = pd.read_csv("../../all_model_results/post_results/test_predictions_prob.csv")

member_level_df = build_strict_24h_member_level_ri_records(
    df=test_result_df,
    id_cols=("SID", "ISO_TIME"),   # 如果你的起报时间列不是 ISO_TIME，要改成实际列名
    forecast_hour_col="Forecast_Hour",
    true_future_wind_col="true_future_wind",
    pred_mu_delta_col="pred_mu_Delta_Wind",
    pred_sigma_delta_col="pred_sigma_Delta_Wind",
    init_wind_col="USA_WIND",
    ri_threshold=30.0,
    n_samples=50,
    random_seed=42,
)

leadtime_result_df = summarize_strict_24h_ri_by_lead_time(member_level_df)

print(leadtime_result_df)

member_level_df.to_csv(
    "./ri_member_level_records_strict24h.csv",
    index=False
)

leadtime_result_df.to_csv(
    "./ri_csi_pss_by_leadtime_strict24h.csv",
    index=False
)