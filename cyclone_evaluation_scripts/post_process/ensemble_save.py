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
from utils import cal_mse, cal_crps, cal_spread, cal_ens
from network import PanguWeather
from data_idx import WeatherDataset40inference
import time
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import autocast
import torch.nn.functional as F
from collections import OrderedDict
from save2nc import save2pt
from network import PanguPerturbWrapper
import datetime

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True


def set_global_seeds(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class VIContainer(nn.Module):
    def __init__(self, base_net_ctor, ctor_kwargs, n_samples):
        super().__init__()
        self.m = base_net_ctor(**ctor_kwargs)
        self.rho = base_net_ctor(**ctor_kwargs)
        self.workhorses = nn.ModuleList(
            [base_net_ctor(**ctor_kwargs) for _ in range(n_samples)]
        )
        self.n_samples = n_samples

    @torch.no_grad()
    def load_vi_state_dict(self, state_dict):
        sd = {
            (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()
        }
        m_sd = {k[2:]: v for k, v in sd.items() if k.startswith("m.")}
        rho_sd = {k[4:]: v for k, v in sd.items() if k.startswith("rho.")}
        self.m.load_state_dict(m_sd, strict=True)
        self.rho.load_state_dict(rho_sd, strict=True)
        for wh in self.workhorses:
            wh.load_state_dict(m_sd, strict=True)

    @torch.no_grad()
    def sample_all_members(self, base_seed, global_rank):
        for i in range(self.n_samples):
            gen = torch.Generator(device=next(self.m.parameters()).device)
            gen.manual_seed(base_seed + global_rank * self.n_samples + i)
            target_net = self.workhorses[i]
            for p_w, p_m, p_rho in zip(
                target_net.parameters(), self.m.parameters(), self.rho.parameters()
            ):
                eps = torch.randn(
                    p_m.shape, dtype=p_m.dtype, device=p_m.device, generator=gen
                )
                s = F.softplus(p_rho)
                p_w.copy_(p_m + s * eps)


def load_and_sample_checkpoint(backbone, perturb_model, ckpt_path, seed, device):
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    perturb_sd = {}
    backbone_mu = {}
    backbone_rho = {}

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        if k.startswith("perturb_model."):
            perturb_sd[k[14:]] = v
        elif k.startswith("vi_model.m."):
            backbone_mu[k[11:]] = v
        elif k.startswith("vi_model.rho."):
            backbone_rho[k[13:]] = v

    perturb_model.load_state_dict(perturb_sd, strict=True)
    print("Perturbation Model loaded.")

    # 2. Backbone Sample & Freeze
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
        attn_drop=0.0,
    ).to(device)

    perturb_model = PanguPerturbWrapper(input_c_upper=10,input_c_surface=11,output_c_upper=args.output_channels,output_c_surface=args.output_channels_surface,).to(device)

    if args.vi_ckpt:
        unique_seed = args.seed + global_rank * 100
        load_and_sample_checkpoint(model, perturb_model, args.vi_ckpt, unique_seed, device)

    model.eval()
    perturb_model.eval()

    topography = torch.from_numpy(np.load('../ERA5_examples/constant_masks/topography.npy').astype(np.float32))
    land_mask = torch.from_numpy(np.load('../ERA5_examples/constant_masks/land_mask.npy').astype(np.float32))
    soil_type = torch.from_numpy(np.load('../ERA5_examples/constant_masks/soil_type.npy').astype(np.float32))

    constant = torch.stack([(topography - topography.mean()) / (topography.std()),land_mask,soil_type], dim=0).to(device=device, dtype=torch.float32)

    mean_upper = torch.load(args.mean_upper).unsqueeze(-1).unsqueeze(-1).to(device)
    std_upper = torch.load(args.std_upper).unsqueeze(-1).unsqueeze(-1).to(device)
    mean_surface = torch.load(args.mean_surface).unsqueeze(-1).unsqueeze(-1).to(device)
    std_surface = torch.load(args.std_surface).unsqueeze(-1).unsqueeze(-1).to(device)

    test_dataset = WeatherDataset40inference(
        utils.parse_list(args.patch_size),
        steps=args.steps,
        filter_time_range_file=args.filter_time_range_file,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    patch_size = utils.parse_list(args.patch_size)
    with torch.inference_mode():
        for idx, batch_data in enumerate(test_loader):
            input_upper, input_surface, now_time, lon_list, lat_list, sid_list = (batch_data)
            current_time = now_time[0]
            date_object = datetime.datetime.strptime(current_time, "%Y-%m-%dT%H:%M")

            month = date_object.month
            year = date_object.year
            dir = "ANONYMOUS_PATH"

            save_dir = f"{dir}{year}-{month:02d}/"

            output_path_list = [save_dir + f"Pangu_{sid[0]}.pt" for sid in sid_list]
            if all(os.path.exists(file) for file in output_path_list):
                if global_rank == 0:
                    print(f"The field information of now time({str(current_time)}) is exist!!!")
                continue

            input_upper = input_upper.to(device, non_blocking=True)
            input_surface = input_surface.to(device, non_blocking=True)

            B, _, Z, H, W = input_upper.shape

            local_members_upper = []
            local_members_surface = []
            start_time = time.time()

            pad_W = (patch_size[2] - W % patch_size[2]) % patch_size[2]
            pad_H = (patch_size[1] - H % patch_size[1]) % patch_size[1]
            pad_Z = (patch_size[0] - Z % patch_size[0]) % patch_size[0]

            for s_idx in range(6):
                curr_input_upper = input_upper.clone()
                curr_input_surface = input_surface.clone()
                input_upper_list = []
                input_surface_list = []

                for step in range(args.steps):
                    input_upper_padded = F.pad(curr_input_upper, (0, pad_W, 0, pad_H, 0, pad_Z))
                    input_surface_padded = F.pad(curr_input_surface, (0, pad_W, 0, pad_H))

                    mu_u, sigma_u, mu_s, sigma_s = perturb_model(input_upper_padded, input_surface_padded, constant, Z, H, W)
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

                    output_upper = out_upper * std_upper + mean_upper
                    output_surface = out_surface * std_surface + mean_surface
                    output_upper = [output_upper[:, 0, 7, :, :],output_upper[:, 2, 10, :, :],]
                    output_surface = [output_surface[:, 1, :, :],output_surface[:, 2, :, :],output_surface[:, 3, :, :],]
                    output_upper = torch.stack(output_upper, dim=1)
                    output_surface = torch.stack(output_surface, dim=1)

                    input_upper_list.append(output_upper)
                    input_surface_list.append(output_surface)

                    curr_input_upper = torch.cat([curr_input_upper[:, -5:, ...], out_upper], dim=1)
                    curr_input_surface = torch.cat([curr_input_surface[:, -4:, ...], out_surface], dim=1)

                input_upper_tensor = torch.stack(input_upper_list, dim=0)
                input_surface_tensor = torch.stack(input_surface_list, dim=0)
                steps, B, C, H, W = input_upper_tensor.shape
                _, _, C_surface, H, W = input_surface_tensor.shape
                input_upper_tensor = input_upper_tensor.view(steps, C, H, W)
                input_surface_tensor = input_surface_tensor.view(steps, C_surface, H, W)
                local_members_upper.append(input_upper_tensor)
                local_members_surface.append(input_surface_tensor)

            local_ens_upper = torch.stack(local_members_upper, dim=1)
            local_ens_surface = torch.stack(local_members_surface, dim=1)

            local_mean_upper = local_ens_upper.mean(dim=1)
            local_mean_surface = local_ens_surface.mean(dim=1)

            global_sum_upper = local_mean_upper.clone()
            global_sum_surface = local_mean_surface.clone()

            dist.all_reduce(global_sum_upper, op=dist.ReduceOp.SUM)
            dist.all_reduce(global_sum_surface, op=dist.ReduceOp.SUM)

            global_mean_upper = global_sum_upper / world_size
            global_mean_surface = global_sum_surface / world_size

            full_ens_upper = global_mean_upper 
            full_ens_surface = global_mean_surface 

            if global_rank == 0:
                save2pt(
                    full_ens_upper,
                    full_ens_surface,
                    now_time,
                    lon_list,
                    lat_list,
                    sid_list,
                )

            if idx % args.log_interval == 0 and global_rank == 0:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                print(f"Iter [{idx + 1}/{len(test_loader)}] Time: {time.time() - start_time:.1f}s | {current_time}")

            del full_ens_upper, full_ens_surface
            del global_mean_upper, global_mean_surface
            del local_ens_upper, local_ens_surface
            del local_members_upper, local_members_surface
            torch.cuda.empty_cache()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    path_dir = "ANONYMOUS_PATH"
    parser = argparse.ArgumentParser("Pangu-Bayes Inference")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--n_samples_per_gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--vi_ckpt",type=str,default="../checkpoint/eu_au_ensemble_forecast.pt")

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
    parser.add_argument("--filter_time_range_file",type=str,default="./train_set.json",)
    parser.add_argument("--log_interval", type=int, default=1)
    args = parser.parse_args()
    print(vars(args))
    main(args)
