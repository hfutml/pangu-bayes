import os
import argparse
import warnings
from datetime import timedelta
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from typing import Tuple, Union, Optional
import utils
from utils import cal_mse, cal_crps, cal_spread,cal_ens
from network import PanguWeather
from data_idx import WeatherDataset40test
import time
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import autocast
from collections import OrderedDict
from network import PanguPerturbWrapper
import pandas as pd
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

def set_global_seeds(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def load_and_sample_checkpoint(backbone, perturb_model, ckpt_path, seed, device):
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    perturb_sd = {}
    backbone_mu = {}
    backbone_rho = {}

    for k, v in state_dict.items():
        if k.startswith('module.'): k = k[7:]
        if k.startswith('perturb_model.'):
            perturb_sd[k[14:]] = v
        elif k.startswith('vi_model.m.'):
            backbone_mu[k[11:]] = v
        elif k.startswith('vi_model.rho.'):
            backbone_rho[k[13:]] = v

    perturb_model.load_state_dict(perturb_sd, strict=True)
    print("Perturbation Model loaded.")

    g = torch.Generator(device=device)
    g.manual_seed(seed)
    print(f"[Rank {dist.get_rank()}] Sampling Backbone with seed {seed}...")
    
    with torch.no_grad():
        for name, param in backbone.named_parameters():
            if name in backbone_mu:
                mu = backbone_mu[name].to(device)
                if name in backbone_rho:
                    rho = backbone_rho[name].to(device)
                    sigma = F.softplus(rho)
                    eps = torch.randn(mu.shape, device=device, generator=g)
                    param.copy_(mu + sigma * eps)
                else:
                    param.copy_(mu)

def weighted_spatial_mean(x, lat_weight):
    lat_weight = lat_weight.to(device=x.device, dtype=x.dtype)

    H = x.shape[-2]
    W = x.shape[-1]

    if lat_weight.numel() == H:
        # Handles [H], [1, H], [H, 1]
        w = lat_weight.reshape(H).view(*([1] * (x.dim() - 2)), H, 1)
        denom = lat_weight.sum() * W

    elif lat_weight.numel() == H * W:
        # Handles real 2D weight [H, W]
        w = lat_weight.reshape(H, W).view(*([1] * (x.dim() - 2)), H, W)
        denom = lat_weight.sum()

    else:
        raise ValueError(
            f"Unsupported lat_weight shape={tuple(lat_weight.shape)}, "
            f"x shape={tuple(x.shape)}, expected numel {H} or {H * W}"
        )

    return (x * w).sum(dim=(-2, -1)) / denom


def mse_all(pred, target, lat_weight):
    ens_mean = pred.mean(dim=1)
    mse = (ens_mean - target).pow(2)
    return weighted_spatial_mean(mse, lat_weight).mean(dim=0)


def spread_all(pred, lat_weight):
    var = pred.var(dim=1, unbiased=True)
    spread = weighted_spatial_mean(var, lat_weight)
    return spread.mean(dim=0)


def crps_all(pred, target, lat_weight):
    member_num = pred.shape[1]

    term1 = (pred - target.unsqueeze(1)).abs().mean(dim=1)

    pred_sorted, _ = pred.sort(dim=1)
    coeff = torch.arange(
        1,
        member_num + 1,
        device=pred.device,
        dtype=pred.dtype
    )
    coeff = 2 * coeff - member_num - 1

    view_shape = [1, member_num] + [1] * (pred.dim() - 2)
    coeff = coeff.view(*view_shape)

    pairwise_mean_abs = 2.0 * (coeff * pred_sorted).sum(dim=1) / (member_num * (member_num-1))
    crps = term1 - 0.5 * pairwise_mean_abs

    return weighted_spatial_mean(crps, lat_weight).mean(dim=0)


def all_gather_members(local_tensor, world_size):
    gathered = [torch.empty_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(gathered, local_tensor)
    return torch.cat(gathered, dim=1)


def evaluate_one_lead(
    local_upper_members_cpu,
    local_surface_members_cpu,
    target_upper_cpu,
    target_surface_cpu,
    lat_weight,
    world_size,
    global_rank,
    device,
    upper_z_chunk,
):
    result = {}

    _, c_upper, z_size, _, _ = target_upper_cpu.shape
    z_chunk = z_size if upper_z_chunk <= 0 else upper_z_chunk

    if global_rank == 0:
        result["upper_mse"] = torch.zeros(c_upper, z_size)
        result["upper_crps"] = torch.zeros(c_upper, z_size)
        result["upper_spread"] = torch.zeros(c_upper, z_size)

    for z0 in range(0, z_size, z_chunk):
        z1 = min(z0 + z_chunk, z_size)

        local_upper = torch.stack(
            [x[:, :, z0:z1, :, :] for x in local_upper_members_cpu],
            dim=1
        ).to(device, non_blocking=True)

        full_upper = all_gather_members(local_upper, world_size)

        if global_rank == 0:
            target_upper = target_upper_cpu[:, :, z0:z1, :, :].to(device, non_blocking=True)

            result["upper_mse"][:, z0:z1] = mse_all(full_upper, target_upper, lat_weight).cpu()
            result["upper_crps"][:, z0:z1] = crps_all(full_upper, target_upper, lat_weight).cpu()
            result["upper_spread"][:, z0:z1] = spread_all(full_upper, lat_weight).cpu()

            del target_upper

        del local_upper, full_upper
        torch.cuda.empty_cache()

    local_surface = torch.stack(local_surface_members_cpu, dim=1).to(device, non_blocking=True)
    full_surface = all_gather_members(local_surface, world_size)

    if global_rank == 0:
        target_surface = target_surface_cpu.to(device, non_blocking=True)

        result["surface_mse"] = mse_all(full_surface, target_surface, lat_weight).cpu()
        result["surface_crps"] = crps_all(full_surface, target_surface, lat_weight).cpu()
        result["surface_spread"] = spread_all(full_surface, lat_weight).cpu()

        del target_surface

    del local_surface, full_surface
    torch.cuda.empty_cache()

    if global_rank == 0:
        return result

    return None

def save_metrics_to_csv(metrics, csv_path):
    rows = []

    upper_names = ["z", "q", "t", "u", "v"]
    surface_names = ["t2m", "u10", "v10", "msl"]

    for lead, lead_metrics in metrics.items():
        for metric_name, values in lead_metrics.items():
            values = values.detach().cpu()

            if metric_name.startswith("upper_"):
                # values shape: [C, Z]
                metric = metric_name.replace("upper_", "")

                for c in range(values.shape[0]):
                    channel_name = upper_names[c] if c < len(upper_names) else f"upper_c{c}"

                    for z in range(values.shape[1]):
                        rows.append({
                            "lead_step": lead,
                            "domain": "upper",
                            "metric": metric,
                            "channel": c,
                            "channel_name": channel_name,
                            "level_index": z,
                            "value": values[c, z].item(),
                        })

            elif metric_name.startswith("surface_"):
                # values shape: [C]
                metric = metric_name.replace("surface_", "")

                for c in range(values.shape[0]):
                    channel_name = surface_names[c] if c < len(surface_names) else f"surface_c{c}"

                    rows.append({
                        "lead_step": lead,
                        "domain": "surface",
                        "metric": metric,
                        "channel": c,
                        "channel_name": channel_name,
                        "level_index": -1,
                        "value": values[c].item(),
                    })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics to: {csv_path}")

def print_all_metrics(metrics):
    for lead, lead_metrics in metrics.items():
        print(f"\n--- Lead Step {lead} ---")
        for name, value in lead_metrics.items():
            print(f"{name}:")
            print(value)

def main(args):
    torch.cuda.empty_cache()
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=7200000))
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    set_global_seeds(args.seed + global_rank)


    model = PanguWeather(
        num_class=args.num_class,
        input_channels=args.input_channels,
        input_channels_surface=args.input_channels_surface,
        output_channels=args.output_channels,
        output_channels_surface=args.output_channels_surface,
        init_channels=args.init_channels,
        input_shape=utils.parse_list(args.input_shape),
        patch_size=utils.parse_list(args.patch_size),
        window_size=utils.parse_list(args.window_size),
        depths=utils.parse_list(args.depths),
        drop_path_rate=0.0,
        num_heads=utils.parse_list(args.num_heads),
        split_size=args.split_size,
        attn_drop=0.0
    ).to(device)

    perturb_model = PanguPerturbWrapper(
        input_c_upper=10,             
        input_c_surface=11,           
        output_c_upper=args.output_channels,  
        output_c_surface=args.output_channels_surface
    ).to(device)

    if args.vi_ckpt:
        unique_seed = args.seed + global_rank * 100
        load_and_sample_checkpoint(model, perturb_model, args.vi_ckpt, unique_seed, device)


    model.eval()
    perturb_model.eval()

    lat_weight = utils.cal_lat_weight().to(device)
    topo_path = os.path.join(args.constant_root, "topography.npy")
    land_path = os.path.join(args.constant_root, "land_mask.npy")
    soil_path = os.path.join(args.constant_root, "soil_type.npy")
    topography = torch.from_numpy(np.load(topo_path).astype(np.float32)).to(device)
    land_mask = torch.from_numpy(np.load(land_path).astype(np.float32)).to(device)
    soil_type = torch.from_numpy(np.load(soil_path).astype(np.float32)).to(device)
    constant = torch.stack([(topography - topography.mean()) / topography.std(), land_mask, soil_type], dim=0)


    mean_upper = torch.load(args.mean_upper).unsqueeze(-1).unsqueeze(-1).to(device)
    std_upper = torch.load(args.std_upper).unsqueeze(-1).unsqueeze(-1).to(device)
    mean_surface = torch.load(args.mean_surface).unsqueeze(-1).unsqueeze(-1).to(device)
    std_surface = torch.load(args.std_surface).unsqueeze(-1).unsqueeze(-1).to(device)



    test_dataset = WeatherDataset40test(utils.parse_list(args.patch_size), is_train=False, steps=args.steps)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,             
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    metrics = None
    patch_size = utils.parse_list(args.patch_size)

    num_eval_batches = (
    min(args.max_eval_batches, len(test_loader))
    if args.max_eval_batches > 0
    else len(test_loader)
    )

    with torch.inference_mode():
        for idx, (input_upper, input_surface, target_upper, target_surface) in enumerate(test_loader):
            if args.max_eval_batches > 0 and idx >= args.max_eval_batches:
                break
            input_upper = input_upper.to(device, non_blocking=True)
            input_surface = input_surface.to(device, non_blocking=True)

            target_upper_cpu = target_upper
            target_surface_cpu = target_surface

            B, T, _, Z, H, W = target_upper_cpu.shape

            pad_W = (patch_size[2] - W % patch_size[2]) % patch_size[2]
            pad_H = (patch_size[1] - H % patch_size[1]) % patch_size[1]
            pad_Z = (patch_size[0] - Z % patch_size[0]) % patch_size[0]

            local_steps_upper = [[] for _ in range(args.steps)]
            local_steps_surface = [[] for _ in range(args.steps)]

            start_time = time.time()

            for s_idx in range(args.au_samples_per_gpu):
                curr_input_upper = input_upper.clone()
                curr_input_surface = input_surface.clone()

                for step in range(args.steps):
                    input_upper_padded = F.pad(curr_input_upper, (0, pad_W, 0, pad_H, 0, pad_Z))
                    input_surface_padded = F.pad(curr_input_surface, (0, pad_W, 0, pad_H))

                    mu_u, sigma_u, mu_s, sigma_s = perturb_model(
                        input_upper_padded,
                        input_surface_padded,
                        constant,
                        Z,
                        H,
                        W
                    )

                    step_seed = (
                        args.seed
                        + idx * 1000
                        + step * 100
                        + s_idx
                        + global_rank * 10000
                    )
                    rng_step = torch.Generator(device=device).manual_seed(step_seed)

                    eps_u = torch.randn(mu_u.shape, device=device, generator=rng_step)
                    eps_s = torch.randn(mu_s.shape, device=device, generator=rng_step)

                    upper_tm1 = curr_input_upper[:, :5]
                    upper_t = curr_input_upper[:, 5:]

                    surface_tm1 = curr_input_surface[:, :4]
                    surface_t = curr_input_surface[:, 4:]

                    upper_t_perturbed = upper_t + (mu_u + sigma_u * eps_u)
                    surface_t_perturbed = surface_t + (mu_s + sigma_s * eps_s)

                    curr_upper_perturbed = torch.cat([upper_tm1, upper_t_perturbed], dim=1)
                    curr_surface_perturbed = torch.cat([surface_tm1, surface_t_perturbed], dim=1)

                    in_u = F.pad(curr_upper_perturbed, (0, pad_W, 0, pad_H, 0, pad_Z))
                    in_s = F.pad(curr_surface_perturbed, (0, pad_W, 0, pad_H))

                    out_upper, out_surface = model(in_u, in_s, constant, Z, H, W)

                    pred_upper = (out_upper * std_upper + mean_upper).detach().cpu()
                    pred_surface = (out_surface * std_surface + mean_surface).detach().cpu()

                    local_steps_upper[step].append(pred_upper)
                    local_steps_surface[step].append(pred_surface)

                    curr_input_upper = torch.cat(
                        [curr_input_upper[:, -5:, ...], out_upper],
                        dim=1
                    )
                    curr_input_surface = torch.cat(
                        [curr_input_surface[:, -4:, ...], out_surface],
                        dim=1
                    )

                    del input_upper_padded, input_surface_padded
                    del mu_u, sigma_u, mu_s, sigma_s
                    del eps_u, eps_s
                    del curr_upper_perturbed, curr_surface_perturbed
                    del in_u, in_s
                    torch.cuda.empty_cache()

            end_time = time.time()
            print(f"[Rank {global_rank}] Batch {idx + 1}/{len(test_loader)} inference time: {end_time - start_time:.2f}s")

            for lead in range(args.steps):
                lead_result = evaluate_one_lead(
                    local_upper_members_cpu=local_steps_upper[lead],
                    local_surface_members_cpu=local_steps_surface[lead],
                    target_upper_cpu=target_upper_cpu[:, lead],
                    target_surface_cpu=target_surface_cpu[:, lead],
                    lat_weight=lat_weight,
                    world_size=world_size,
                    global_rank=global_rank,
                    device=device,
                    upper_z_chunk=args.upper_z_chunk,
                )

                if global_rank == 0:
                    if metrics is None:
                        metrics = {
                            step_id: {
                                name: torch.zeros_like(value)
                                for name, value in lead_result.items()
                            }
                            for step_id in range(1, args.steps + 1)
                        }

                    for name, value in lead_result.items():
                        metrics[lead + 1][name] += value / num_eval_batches

            del local_steps_upper, local_steps_surface
            del input_upper, input_surface
            del target_upper_cpu, target_surface_cpu
            torch.cuda.empty_cache()

    if global_rank == 0:
        if metrics is None:
            print("\nNo metrics were computed.")
        else:
            print("\n--- Final Metrics: All Lead Steps, All Channels ---")
            print_all_metrics(metrics)
            save_metrics_to_csv(metrics, args.metrics_csv)
    dist.barrier()
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference (48-member)")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--metrics_csv", type=str, default="metrics_ec0_era5.csv")
    parser.add_argument("--upper_z_chunk", type=int, default=13)
    parser.add_argument("--max_eval_batches", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--au_samples_per_gpu", type=int, default=6)
    parser.add_argument("--seed", type=int, default=12534)
    parser.add_argument("--vi_ckpt", type=str, default="/home/u2024110474/xiongxinlei/NMI/checkpoint/eu_au_ensemble_forecast_hres.pt")
    parser.add_argument("--constant_root", type=str,default="../ERA5_examples/constant_masks")
    parser.add_argument("--mean_upper", type=str, default="../ERA5_examples/statistic/upper_mean_40.pt")
    parser.add_argument("--std_upper", type=str, default="../ERA5_examples/statistic/upper_std_40.pt")
    parser.add_argument("--mean_surface", type=str, default="../ERA5_examples/statistic/surface_mean_40.pt")
    parser.add_argument("--std_surface", type=str, default="../ERA5_examples/statistic/surface_std_40.pt")
    parser.add_argument("--depths", type=str, default="8_24")
    parser.add_argument("--patch_size", type=str, default="2_4_4")
    parser.add_argument("--window_size", type=str, default="2_6_12")
    parser.add_argument("--num_heads", type=str, default="6_12")
    parser.add_argument("--drop_path_rate", type=float, default=0.0)
    parser.add_argument("--attn_drop", type=float, default=0.0)
    parser.add_argument("--split_size", type=int, default=1)
    parser.add_argument("--num_class", type=int, default=2)
    parser.add_argument("--input_channels", type=int, default=10)
    parser.add_argument("--input_channels_surface", type=int, default=11)
    parser.add_argument("--output_channels", type=int, default=5)
    parser.add_argument("--output_channels_surface", type=int, default=4)
    parser.add_argument("--init_channels", type=int, default=192)
    parser.add_argument("--input_shape", type=str, default="13_721_1440")
    args = parser.parse_args()
    main(args)

