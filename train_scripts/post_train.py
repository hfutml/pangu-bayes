import os
import math
import random
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    train_path: str = "../evaluation_scripts/post_process/dataset/statistical_variables_train.csv"
    test_path: str = "../evaluation_scripts/post_process/dataset/statistical_variables_test.csv"
    save_dir: str = "../all_model_results/post_results"

    seed: int = 42
    batch_size: int = 32
    num_workers: int = 0
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-4

    hidden_dim: int = 44
    num_hidden_layers: int = 6
    dropout: float = 0.20
    activation: str = "gelu"

    use_latlon: bool = True
    convert_p0min_to_hpa: bool = True

    eval_every: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def get_feature_columns(use_latlon: bool = True) -> List[str]:
    base_cols = [
        "USA_WIND",
        "USA_PRES",
        "Forecast_Hour",
        "V10_max",
        "P0_min",
        "V10_range",
        "P0_range",
        "z500_min",
        "t850_range",
    ]
    if use_latlon:
        return ["LAT", "LON"] + base_cols
    return base_cols


TARGET_COLS = ["Target_Delta_Wind", "Target_Delta_Pres"]


def load_one_dataframe(path: str, cfg: Config) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}")

    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path)
    elif path.endswith(".txt") or path.endswith(".tsv"):
        df = pd.read_csv(path, sep="\t")
    else:
        raise ValueError(f"不支持的文件格式: {path}")

    df = df.copy()
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"])

    if cfg.convert_p0min_to_hpa:
        if "P0_min" in df.columns and df["P0_min"].mean() > 2000:
            df["P0_min"] = df["P0_min"] / 100.0
        if "P0_range" in df.columns and df["P0_range"].mean() > 2000:
            df["P0_range"] = df["P0_range"] / 100.0

    feature_cols = get_feature_columns(cfg.use_latlon)
    required_cols = feature_cols + TARGET_COLS

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{path} 缺少必要列: {missing_cols}")

    df = df.dropna(subset=required_cols).reset_index(drop=True)
    return df


class TCStatDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str],
        x_scaler: StandardScaler = None,
        y_scaler: StandardScaler = None,
        fit_scaler: bool = False,
    ):
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.target_cols = target_cols

        X = self.df[self.feature_cols].values.astype(np.float32)
        y = self.df[self.target_cols].values.astype(np.float32)

        if x_scaler is None:
            x_scaler = StandardScaler()
        if y_scaler is None:
            y_scaler = StandardScaler()

        if fit_scaler:
            X = x_scaler.fit_transform(X)
            y = y_scaler.fit_transform(y)
        else:
            X = x_scaler.transform(X)
            y = y_scaler.transform(y)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ProbANN(nn.Module):
    def __init__(
        self,
        num_scalars: int,
        hidden_dim: int = 64,
        num_hidden_layers: int = 5,
        dropout: float = 0.15,
        activation: str = "gelu",
    ):
        super().__init__()

        if activation.lower() == "relu":
            act_layer = nn.ReLU
        elif activation.lower() == "tanh":
            act_layer = nn.Tanh

        elif activation.lower() == "hardswish":
            act_layer = nn.Hardswish
            
        else:
            act_layer = nn.GELU

        layers = []
        in_dim = num_scalars
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_layer())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 4))
        self.layers = nn.Sequential(*layers)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = torch.squeeze(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        out = self.layers(x)

        mu_w = out[:, 0:1]
        sigma_w = self.softplus(out[:, 1:2]) + 1e-6
        mu_p = out[:, 2:3]
        sigma_p = self.softplus(out[:, 3:4]) + 1e-6

        return torch.cat([mu_w, sigma_w, mu_p, sigma_p], dim=1)


def gaussian_crps_torch(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = (y - mu) / sigma
    sqrt_2 = math.sqrt(2.0)
    sqrt_pi_inv = 1.0 / math.sqrt(math.pi)

    phi = torch.exp(-0.5 * z ** 2) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + torch.erf(z / sqrt_2))

    crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - sqrt_pi_inv)
    return crps


def probabilistic_crps_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mu_w = pred[:, 0:1]
    sigma_w = pred[:, 1:2]
    mu_p = pred[:, 2:3]
    sigma_p = pred[:, 3:4]

    y_w = target[:, 0:1]
    y_p = target[:, 1:2]

    crps_w = gaussian_crps_torch(mu_w, sigma_w, y_w)
    crps_p = gaussian_crps_torch(mu_p, sigma_p, y_p)

    loss = (crps_w.mean() + crps_p.mean()) / 2.0
    return loss


def gaussian_crps_numpy(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = (y - mu) / sigma
    phi = np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)
    Phi = 0.5 * (1.0 + np.vectorize(math.erf)(z / np.sqrt(2.0)))
    crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))
    return crps


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = probabilistic_crps_loss(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    total_loss = 0.0

    all_preds = []
    all_targets = []

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = probabilistic_crps_loss(pred, y)

        total_loss += loss.item() * X.size(0)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    mean_loss = total_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return mean_loss, all_preds, all_targets

def inverse_transform_probabilistic_predictions(pred_std: np.ndarray, y_scaler: StandardScaler) -> Dict[str, np.ndarray]:
    y_mean = y_scaler.mean_
    y_scale = y_scaler.scale_

    mu_w_std = pred_std[:, 0]
    sigma_w_std = pred_std[:, 1]
    mu_p_std = pred_std[:, 2]
    sigma_p_std = pred_std[:, 3]

    mu_w = mu_w_std * y_scale[0] + y_mean[0]
    sigma_w = sigma_w_std * y_scale[0]

    mu_p = mu_p_std * y_scale[1] + y_mean[1]
    sigma_p = sigma_p_std * y_scale[1]

    sigma_w = np.clip(sigma_w, 1e-6, None)
    sigma_p = np.clip(sigma_p, 1e-6, None)

    return {
        "mu_w": mu_w,
        "sigma_w": sigma_w,
        "mu_p": mu_p,
        "sigma_p": sigma_p,
    }


def inverse_transform_targets(target_std: np.ndarray, y_scaler: StandardScaler) -> np.ndarray:
    return y_scaler.inverse_transform(target_std)

def evaluate_overall_metrics(
    true_y: np.ndarray,
    pred_mu_w: np.ndarray,
    pred_sigma_w: np.ndarray,
    pred_mu_p: np.ndarray,
    pred_sigma_p: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    true_w = true_y[:, 0]
    true_p = true_y[:, 1]

    mae_w = mean_absolute_error(true_w, pred_mu_w)
    mae_p = mean_absolute_error(true_p, pred_mu_p)

    crps_w = gaussian_crps_numpy(pred_mu_w, pred_sigma_w, true_w).mean()
    crps_p = gaussian_crps_numpy(pred_mu_p, pred_sigma_p, true_p).mean()

    results = {
        f"{prefix}Delta_Wind_MAE": mae_w,
        f"{prefix}Delta_Pres_MAE": mae_p,
        f"{prefix}Delta_Wind_CRPS": crps_w,
        f"{prefix}Delta_Pres_CRPS": crps_p,
        f"{prefix}Mean_MAE": 0.5 * (mae_w + mae_p),
        f"{prefix}Mean_CRPS": 0.5 * (crps_w + crps_p),
    }
    return results


def evaluate_by_forecast_hour(result_df: pd.DataFrame, save_path: str = None) -> pd.DataFrame:
    rows = []

    for fh, sub in result_df.groupby("Forecast_Hour"):
        true_w = sub["true_Delta_Wind"].values
        true_p = sub["true_Delta_Pres"].values

        mu_w = sub["pred_mu_Delta_Wind"].values
        sigma_w = sub["pred_sigma_Delta_Wind"].values
        mu_p = sub["pred_mu_Delta_Pres"].values
        sigma_p = sub["pred_sigma_Delta_Pres"].values

        row = {
            "Forecast_Hour": fh,
            "Count": len(sub),

            "Delta_Wind_MAE": mean_absolute_error(true_w, mu_w),
            "Delta_Pres_MAE": mean_absolute_error(true_p, mu_p),

            "Delta_Wind_CRPS": gaussian_crps_numpy(mu_w, sigma_w, true_w).mean(),
            "Delta_Pres_CRPS": gaussian_crps_numpy(mu_p, sigma_p, true_p).mean(),
        }
        rows.append(row)

    metric_df = pd.DataFrame(rows).sort_values("Forecast_Hour").reset_index(drop=True)

    if save_path is not None:
        metric_df.to_csv(save_path, index=False)

    return metric_df


def main():
    cfg = Config()
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    print(f"Using device: {cfg.device}")

    feature_cols = get_feature_columns(cfg.use_latlon)

    train_df = load_one_dataframe(cfg.train_path, cfg)
    test_df = load_one_dataframe(cfg.test_path, cfg)

    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    print(f"Target columns: {TARGET_COLS}")

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    train_dataset = TCStatDataset(
        train_df,
        feature_cols,
        TARGET_COLS,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        fit_scaler=True,
    )
    test_dataset = TCStatDataset(
        test_df,
        feature_cols,
        TARGET_COLS,
        x_scaler=train_dataset.x_scaler,
        y_scaler=train_dataset.y_scaler,
        fit_scaler=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    model = ProbANN(
        num_scalars=len(feature_cols),
        hidden_dim=cfg.hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        dropout=cfg.dropout,
        activation=cfg.activation,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    best_test_crps = float("inf")
    best_epoch = -1
    history = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, cfg.device)

        row = {
            "epoch": epoch,
            "train_loss_std_space_crps": train_loss,
        }

        if epoch % cfg.eval_every == 0:
            test_loss_std, test_pred_std, test_true_std = predict_all(model, test_loader, cfg.device)
            scheduler.step()

            pred_phys = inverse_transform_probabilistic_predictions(test_pred_std, train_dataset.y_scaler)
            true_phys = inverse_transform_targets(test_true_std, train_dataset.y_scaler)

            test_metrics = evaluate_overall_metrics(
                true_y=true_phys,
                pred_mu_w=pred_phys["mu_w"],
                pred_sigma_w=pred_phys["sigma_w"],
                pred_mu_p=pred_phys["mu_p"],
                pred_sigma_p=pred_phys["sigma_p"],
                prefix="test_",
            )

            current_lr = optimizer.param_groups[0]["lr"]

            row["test_loss_std_space_crps"] = test_loss_std
            row["learning_rate"] = current_lr
            row.update(test_metrics)


            print(
                f"Epoch [{epoch:03d}/{cfg.epochs}] "
                f"lr={current_lr:.8f} "
                f"train_crps(std)={train_loss:.6f} "
                f"test_mean_crps={test_metrics['test_Mean_CRPS']:.4f} "
                f"test_dw_mae={test_metrics['test_Delta_Wind_MAE']:.4f} "
                f"test_dp_mae={test_metrics['test_Delta_Pres_MAE']:.4f}"
            )

            if test_metrics["test_Mean_CRPS"] < best_test_crps:
                best_test_crps = test_metrics["test_Mean_CRPS"]
                best_epoch = epoch

                ckpt = {
                    "model_state_dict": model.state_dict(),
                    "feature_cols": feature_cols,
                    "target_cols": TARGET_COLS,
                    "config": cfg.__dict__,
                    "best_test_crps": best_test_crps,
                    "best_epoch": best_epoch,
                    "x_scaler_mean": train_dataset.x_scaler.mean_,
                    "x_scaler_scale": train_dataset.x_scaler.scale_,
                    "y_scaler_mean": train_dataset.y_scaler.mean_,
                    "y_scaler_scale": train_dataset.y_scaler.scale_,
                }
                torch.save(ckpt, os.path.join(cfg.save_dir, "best_prob_ann.pt"))
            else:
                row["learning_rate"] = optimizer.param_groups[0]["lr"]

        history.append(row)

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(cfg.save_dir, "training_history.csv"), index=False)

    best_ckpt = torch.load(os.path.join(cfg.save_dir, "best_prob_ann.pt"), map_location=cfg.device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_loss_std, test_pred_std, test_true_std = predict_all(model, test_loader, cfg.device)

    pred_phys = inverse_transform_probabilistic_predictions(test_pred_std, train_dataset.y_scaler)
    true_phys = inverse_transform_targets(test_true_std, train_dataset.y_scaler)

    test_metrics = evaluate_overall_metrics(
        true_y=true_phys,
        pred_mu_w=pred_phys["mu_w"],
        pred_sigma_w=pred_phys["sigma_w"],
        pred_mu_p=pred_phys["mu_p"],
        pred_sigma_p=pred_phys["sigma_p"],
        prefix="test_",
    )

    print("\n===== Overall Test Results =====")
    print(f"best_epoch = {best_epoch}")
    print(f"test_loss_std_space_crps = {test_loss_std:.6f}")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.6f}")

    test_result_df = test_df.copy().reset_index(drop=True)

    test_result_df["true_Delta_Wind"] = true_phys[:, 0]
    test_result_df["true_Delta_Pres"] = true_phys[:, 1]

    test_result_df["pred_mu_Delta_Wind"] = pred_phys["mu_w"]
    test_result_df["pred_sigma_Delta_Wind"] = pred_phys["sigma_w"]
    test_result_df["pred_mu_Delta_Pres"] = pred_phys["mu_p"]
    test_result_df["pred_sigma_Delta_Pres"] = pred_phys["sigma_p"]

    test_result_df["pred_future_wind"] = test_result_df["USA_WIND"] + test_result_df["pred_mu_Delta_Wind"]
    test_result_df["pred_future_pres"] = test_result_df["USA_PRES"] + test_result_df["pred_mu_Delta_Pres"]
    test_result_df["true_future_wind"] = test_result_df["USA_WIND"] + test_result_df["true_Delta_Wind"]
    test_result_df["true_future_pres"] = test_result_df["USA_PRES"] + test_result_df["true_Delta_Pres"]

    test_result_df.to_csv(os.path.join(cfg.save_dir, "test_predictions_prob.csv"), index=False)

    # 按 forecast hour 统计
    fh_metric_df = evaluate_by_forecast_hour(
        test_result_df,
        save_path=os.path.join(cfg.save_dir, "metrics_by_forecast_hour_prob.csv")
    )

    print("\n===== Metrics by Forecast_Hour =====")
    print(fh_metric_df.to_string(index=False))

    print(f"\nOutputs saved in: {cfg.save_dir}")


if __name__ == "__main__":
    main()

