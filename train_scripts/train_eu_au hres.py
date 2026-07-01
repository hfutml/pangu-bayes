import os
import argparse
import warnings
from datetime import timedelta
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from torch.utils.checkpoint import checkpoint

import numpy as np
from network import PanguPerturbWrapper
from network import PanguWeather
from data_idx import WeatherDataset40hresbuff
import utils

warnings.filterwarnings("ignore")


def compute_fair_crps_loss(preds, target, lat_weight, channel_weights):
    M = preds.shape[0]

    view_shape = [1] * target.ndim
    view_shape[1] = -1
    w = channel_weights.view(*view_shape)

    diff_truth = torch.abs(preds - target.unsqueeze(0))
    term_accuracy = (diff_truth * lat_weight * w).mean()

    if M > 1:
        sum_diff = 0.0
        for i in range(M):
            for j in range(i + 1, M):
                diff = torch.abs(preds[i] - preds[j])
                sum_diff += (diff * lat_weight * w).mean()

        term_spread = sum_diff / (M * (M - 1))
    else:
        term_spread = torch.tensor(0.0, device=target.device)

    loss = term_accuracy - term_spread
    return loss, term_accuracy, term_spread


class FrozenVISampler(nn.Module):
    def __init__(self, base_net_ctor, ctor_kwargs):
        super().__init__()

        self.m = base_net_ctor(**ctor_kwargs)
        self.rho = base_net_ctor(**ctor_kwargs)
        self.runner = base_net_ctor(**ctor_kwargs)

        for p in self.m.parameters():
            p.requires_grad = False
        for p in self.rho.parameters():
            p.requires_grad = False
        for p in self.runner.parameters():
            p.requires_grad = False

    def to(self, *args, **kwargs):
        self.runner.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def load_vi_state_dict(self, state_dict):
        print("Loading VI weights...")

        sd = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }

        m_sd = {k[2:]: v for k, v in sd.items() if k.startswith("m.")}
        rho_sd = {k[4:]: v for k, v in sd.items() if k.startswith("rho.")}

        self.m.load_state_dict(m_sd, strict=True)
        self.rho.load_state_dict(rho_sd, strict=True)
        self.runner.load_state_dict(m_sd, strict=True)

        print("Frozen VI weights loaded.")

    @torch.no_grad()
    def resample_weights(self, seed):
        device = next(self.runner.parameters()).device

        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

        for p_runner, p_m, p_rho in zip(
            self.runner.parameters(),
            self.m.parameters(),
            self.rho.parameters(),
        ):
            t_m = p_m.to(device, non_blocking=True)
            t_rho = p_rho.to(device, non_blocking=True)

            eps = torch.randn(
                t_m.shape,
                dtype=t_m.dtype,
                device=device,
                generator=gen,
            )

            sigma = F.softplus(t_rho)
            p_runner.copy_(t_m + sigma * eps)

    def forward(self, *args, **kwargs):
        return self.runner(*args, **kwargs)


class VI_Pangu_AR_Wrapper(nn.Module):
    def __init__(
        self,
        base_net_ctor,
        base_net_kwargs,
        output_channels,
        output_channels_surface,
        input_c_upper=10,
        input_c_surface=11,
    ):
        super().__init__()

        self.vi_model = FrozenVISampler(base_net_ctor, base_net_kwargs)

        self.out_c = output_channels
        self.out_c_s = output_channels_surface
        self.patch_size = [2, 4, 4]

        self.perturb_model = PanguPerturbWrapper(
            input_c_upper=input_c_upper,
            input_c_surface=input_c_surface,
            output_c_upper=output_channels,
            output_c_surface=output_channels_surface,
        )

    def load_backbone_ckpt(self, path):
        print(f"Loading backbone from: {path}")
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        self.vi_model.load_vi_state_dict(state_dict)

    def forward(self, inputs, targets_seq, loss_fn_dict, ar_steps=1):
        input_upper, input_surface, constant = inputs
        B = input_upper.shape[0]

        curr_upper = input_upper
        curr_surface = input_surface

        total_loss = 0.0
        log_acc_u = torch.tensor(0.0, device=input_upper.device)
        log_spread_u = torch.tensor(0.0, device=input_upper.device)

        def run_model(u, s, c, z, h, w):
            return self.vi_model(u, s, c, z, h, w)

        for step in range(ar_steps):
            target_u_step, target_s_step = targets_seq[step]

            tgt_Z, tgt_H, tgt_W = target_u_step.shape[-3:]

            pad_W = (self.patch_size[2] - tgt_W % self.patch_size[2]) % self.patch_size[2]
            pad_H = (self.patch_size[1] - tgt_H % self.patch_size[1]) % self.patch_size[1]
            pad_Z = (self.patch_size[0] - tgt_Z % self.patch_size[0]) % self.patch_size[0]

            curr_upper_pad = F.pad(curr_upper, (0, pad_W, 0, pad_H, 0, pad_Z))
            curr_surface_pad = F.pad(curr_surface, (0, pad_W, 0, pad_H))

            mu_u, sigma_u, mu_s, sigma_s = self.perturb_model(
                curr_upper_pad,
                curr_surface_pad,
                constant,
                tgt_Z,
                tgt_H,
                tgt_W,
            )

            upper_tm1 = curr_upper[:, :5]
            upper_t = curr_upper[:, 5:]

            surface_tm1 = curr_surface[:, :4]
            surface_t = curr_surface[:, 4:]

            eps_u = torch.randn_like(mu_u)
            eps_s = torch.randn_like(mu_s)

            upper_t_perturbed = upper_t + (mu_u + sigma_u * eps_u)
            surface_t_perturbed = surface_t + (mu_s + sigma_s * eps_s)

            curr_upper_perturbed = torch.cat(
                [upper_tm1, upper_t_perturbed],
                dim=1,
            )
            curr_surface_perturbed = torch.cat(
                [surface_tm1, surface_t_perturbed],
                dim=1,
            )

            in_u = F.pad(curr_upper_perturbed, (0, pad_W, 0, pad_H, 0, pad_Z))
            in_s = F.pad(curr_surface_perturbed, (0, pad_W, 0, pad_H))

            out_u, out_s = checkpoint(
                run_model,
                in_u,
                in_s,
                constant,
                tgt_Z,
                tgt_H,
                tgt_W,
                use_reentrant=False,
            )

            gathered_u = dist_nn.all_gather(out_u)
            gathered_s = dist_nn.all_gather(out_s)

            if isinstance(gathered_u, (tuple, list)):
                ensemble_u = torch.stack(gathered_u, dim=0)
                ensemble_s = torch.stack(gathered_s, dim=0)
            else:
                world_size = dist.get_world_size()
                ensemble_u = gathered_u.view(world_size, B, *out_u.shape[1:])
                ensemble_s = gathered_s.view(world_size, B, *out_s.shape[1:])

            lw = loss_fn_dict["lat_weight"]
            w_u = loss_fn_dict["loss_weights"]["upper"]
            w_s = loss_fn_dict["loss_weights"]["surface"]

            loss_u, acc_u, spread_u = compute_fair_crps_loss(
                ensemble_u,
                target_u_step,
                lw,
                w_u,
            )
            loss_s, _, _ = compute_fair_crps_loss(
                ensemble_s,
                target_s_step,
                lw,
                w_s,
            )

            total_loss += loss_u + 0.25 * loss_s

            if step == ar_steps - 1:
                log_acc_u = acc_u.detach()
                log_spread_u = spread_u.detach()

            if step < ar_steps - 1:
                hist_u = curr_upper[:, self.out_c:, ...]
                hist_s = curr_surface[:, self.out_c_s:, ...]

                curr_upper = torch.cat([hist_u, out_u], dim=1)
                curr_surface = torch.cat([hist_s, out_s], dim=1)

        return total_loss / ar_steps, {
            "acc_u": log_acc_u,
            "spread_u": log_spread_u,
        }


def strip_module_prefix(state_dict):
    return {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def main(args):
    utils.set_seed(args.seed)

    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=7200000),
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if global_rank == 0:
        print("Start Frozen-VI Input-Perturb Training")

    pangu_kwargs = dict(
        num_class=args.num_class,
        input_channels=args.input_channels,
        input_channels_surface=args.input_channels_surface,
        output_channels=args.output_channels,
        output_channels_surface=args.output_channels_surface,
        init_channels=args.init_channels,
        depths=utils.parse_list(args.depths),
        window_size=utils.parse_list(args.window_size),
        patch_size=utils.parse_list(args.patch_size),
        num_heads=utils.parse_list(args.num_heads),
        drop_path_rate=args.drop_path_rate,
        split_size=args.split_size,
        attn_drop=args.attn_drop,
        influence_length=10,
        influence_length_surface=8,
    )

    vi_model = VI_Pangu_AR_Wrapper(
        base_net_ctor=PanguWeather,
        base_net_kwargs=pangu_kwargs,
        output_channels=args.output_channels,
        output_channels_surface=args.output_channels_surface,
        input_c_upper=10,
        input_c_surface=11,
    ).to(device)

    start_step = args.start_step
    resume_optimizer_state = None

    if args.resume_ckpt:
        if global_rank == 0:
            print(f"Loading resume checkpoint: {args.resume_ckpt}")

        ckpt = torch.load(args.resume_ckpt, map_location="cpu")

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            start_step = int(ckpt.get("global_step", args.start_step))
            resume_optimizer_state = ckpt.get("optimizer_state_dict", None)
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            start_step = int(ckpt.get("global_step", args.start_step))
            resume_optimizer_state = ckpt.get("optimizer_state_dict", None)
        else:
            # Compatible with old checkpoints saved by:
            # torch.save(vi_model.module.state_dict(), path)
            state_dict = ckpt
            start_step = args.start_step

        state_dict = strip_module_prefix(state_dict)
        vi_model.load_state_dict(state_dict, strict=True)

        if global_rank == 0:
            print(f"Loaded resume checkpoint. start_step={start_step}")

    elif args.vi_ckpt:
        vi_model.load_backbone_ckpt(args.vi_ckpt)

    vi_model = DistributedDataParallel(
        vi_model,
        device_ids=[local_rank],
        find_unused_parameters=False,
    )

    # Fixed 8 models: each rank samples once, then keeps the same weights.
    vi_model.module.vi_model.resample_weights(args.seed + global_rank)

    trainable_params = filter(
        lambda p: p.requires_grad,
        vi_model.parameters(),
    )

    AR1_STEPS = 5000
    AR_STAGE_STEPS = 3000
    MAX_AR_STEPS = 4
    TOTAL_TRAIN_STEPS = AR1_STEPS + (MAX_AR_STEPS - 1) * AR_STAGE_STEPS
    FIXED_LR = 3e-6

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=FIXED_LR,
        weight_decay=args.weight_decay,
    )

    if resume_optimizer_state is not None:
        optimizer.load_state_dict(resume_optimizer_state)
        move_optimizer_state_to_device(optimizer, device)

        if global_rank == 0:
            print("Loaded optimizer state.")

    train_dataset = WeatherDataset40buff(
        patch_size=utils.parse_list(args.patch_size),
        is_train=True,
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=1,
        rank=0,
        shuffle=True,
    )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
    )

    if args.num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=1,
            timeout=600,
        )

    train_loader = DataLoader(
        train_dataset,
        **loader_kwargs,
    )

    if global_rank == 0:
        print(f"len(train_loader) = {len(train_loader)}")

    utils.set_seed(args.seed + global_rank)

    lat_weight = utils.cal_lat_weight().to(device, dtype=torch.float32)

    loss_fn_dict = {
        "lat_weight": lat_weight,
        "loss_weights": {
            "upper": torch.tensor(
                [3.0, 0.6, 1.5, 0.77, 0.54],
                device=device,
            ),
            "surface": torch.tensor(
                [3.0, 0.77, 0.66, 1.5],
                device=device,
            ),
        },
    }

    topography = torch.from_numpy(
        np.load(os.path.join(args.constant_root, "topography.npy")).astype(np.float32)
    )
    land_mask = torch.from_numpy(
        np.load(os.path.join(args.constant_root, "land_mask.npy")).astype(np.float32)
    )
    soil_type = torch.from_numpy(
        np.load(os.path.join(args.constant_root, "soil_type.npy")).astype(np.float32)
    )

    constant = torch.stack(
        [
            (topography - topography.mean()) / topography.std(),
            land_mask,
            soil_type,
        ],
        dim=0,
    ).to(device=device, dtype=torch.float32)

    vi_model.module.perturb_model.train()
    vi_model.module.vi_model.eval()

    if global_rank == 0:
        print("Mode: AR curriculum training (AR1 -> AR4)")
        print(
            f"AR schedule: steps 1-{AR1_STEPS} => AR1, "
            f"then every {AR_STAGE_STEPS} steps increase by 1 until AR{MAX_AR_STEPS}; "
            f"training stops after {TOTAL_TRAIN_STEPS} steps."
        )
        print(f"Resume from global_step={start_step}")

    global_step = start_step
    epoch = start_step // max(len(train_loader), 1)

    while global_step < TOTAL_TRAIN_STEPS:
        train_sampler.set_epoch(epoch)

        if global_rank == 0:
            print(f"\nStarting epoch {epoch + 1}, global_step={global_step}")

        for batch_data in train_loader:
            if global_step >= TOTAL_TRAIN_STEPS:
                break

            global_step += 1

            if global_step <= AR1_STEPS:
                curr_ar_steps = 1
            elif global_step <= AR1_STEPS + AR_STAGE_STEPS:
                curr_ar_steps = 2
            elif global_step <= AR1_STEPS + 2 * AR_STAGE_STEPS:
                curr_ar_steps = 3
            else:
                curr_ar_steps = 4

            (
                input_u,
                input_s,
                _,
                _,
                target_u,
                target_s,
                data_idx,
            ) = batch_data

            input_u = input_u.to(device, non_blocking=True)
            input_s = input_s.to(device, non_blocking=True)

            targets_seq = [
                (
                    target_u.to(device, non_blocking=True),
                    target_s.to(device, non_blocking=True),
                )
            ]

            if curr_ar_steps > 1:
                with torch.no_grad():
                    for s in range(1, curr_ar_steps):
                        next_u_list = []
                        next_s_list = []

                        for b in range(input_u.shape[0]):
                            curr_id = data_idx[b].item()

                            try:
                                t_u, t_s = train_dataset.get_target(curr_id + s)
                            except (IndexError, KeyError, ValueError) as error:
                                raise RuntimeError(
                                    f"Missing AR target: "
                                    f"global_step={global_step}, "
                                    f"sample={curr_id}, "
                                    f"offset={s}"
                                ) from error

                            next_u_list.append(t_u)
                            next_s_list.append(t_s)

                        targets_seq.append(
                            (
                                torch.stack(next_u_list).to(device, non_blocking=True),
                                torch.stack(next_s_list).to(device, non_blocking=True),
                            )
                        )

            actual_ar_steps = len(targets_seq)

            if actual_ar_steps != curr_ar_steps:
                raise RuntimeError(
                    f"actual_ar_steps={actual_ar_steps}, "
                    f"curr_ar_steps={curr_ar_steps}"
                )

            optimizer.zero_grad(set_to_none=True)

            with autocast(dtype=torch.bfloat16):
                loss, log_info = vi_model(
                    inputs=(input_u, input_s, constant),
                    targets_seq=targets_seq,
                    loss_fn_dict=loss_fn_dict,
                    ar_steps=actual_ar_steps,
                )

            loss.backward()
            optimizer.step()

            if global_rank == 0 and (
                global_step % args.log_interval == 0
                or global_step == TOTAL_TRAIN_STEPS
            ):
                current_time = time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(),
                )

                print(
                    f"Epoch {epoch + 1} | "
                    f"Step {global_step}/{TOTAL_TRAIN_STEPS} | "
                    f"AR: {actual_ar_steps} | "
                    f"Loss: {loss.item():.4f} | "
                    f"L1_upper: {log_info['acc_u'].item():.4f} | "
                    f"Spread_upper: {log_info['spread_u'].item():.4f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                    f"Time: {current_time}"
                )

            dist.barrier()

            if global_rank == 0 and (
                global_step % 4000 == 0
                or global_step == TOTAL_TRAIN_STEPS
            ):
                save_path = f"perturb_ar_curriculum_step_{global_step}.pt"

                print(f"Saving checkpoint to {save_path}")

                torch.save(
                    {
                        "state_dict": vi_model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                        "seed": args.seed,
                        "world_size": dist.get_world_size(),
                        "fixed_lr": FIXED_LR,
                    },
                    save_path,
                )

        epoch += 1

    if global_rank == 0:
        print(
            f"Training completed: "
            f"{global_step} steps, {epoch} epochs."
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Input-Perturb Training")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12534)

    parser.add_argument("--vi_ckpt",type=str,default="../checkpoint/eu_ensemble_forecast.pt",)
    parser.add_argument("--resume_ckpt",
        type=str,
        default="",
    )
    parser.add_argument(
        "--start_step",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--constant_root",
        type=str,
        default="../ERA5_examples/constant_masks",
    )

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

    args = parser.parse_args()
    main(args)