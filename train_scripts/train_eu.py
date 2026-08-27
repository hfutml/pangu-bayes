import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from utils import cal_mae, cal_rmse, clip_grad_norm_, cal_spread, cal_mse
import os
import copy
import time
import argparse
import numpy as np
import warnings
from datetime import timedelta
from tqdm import tqdm
from network import PanguWeather 
from data_idx import WeatherDataset40, WeatherDataset40test
import utils 
from collections import OrderedDict
warnings.filterwarnings("ignore")


class VI_Model(nn.Module):
    def __init__(self, net_init, ND, prior_sig=1.0):
        super().__init__()
        self.m = copy.deepcopy(net_init) 
        self.rho = copy.deepcopy(net_init) 
        with torch.no_grad():
            for p in self.rho.parameters():
                p.copy_(torch.full_like(p, -10.0))

        self.ND = ND
        self.prior_sig = prior_sig

    def forward(self, workhorse_net, prior_net, inputs, targets, loss_fn_dict, beta):
        input_upper, input_surface, constant = inputs
        target_upper, target_surface = targets
        criterion = loss_fn_dict['criterion']
        lat_weight = loss_fn_dict['lat_weight']
        loss_weights = loss_fn_dict['loss_weights']
        upper_surface_weights = loss_fn_dict['upper_surface_weights']
        with torch.no_grad():
            m_params = self.m.parameters()
            rho_params = self.rho.parameters()
            for p, p_m, p_rho in zip(workhorse_net.parameters(), m_params, rho_params):
                eps = torch.randn_like(p)
                s = F.softplus(p_rho)
                p.copy_(p_m + s * eps) 

        with autocast(dtype=torch.bfloat16):
            B, _, Z_in, H_in, W_in = target_upper.shape
            output_upper, output_surface = workhorse_net(input_upper, input_surface, constant, Z_in, H_in, W_in)
            
            loss_matrix_upper = criterion(output_upper, target_upper) * lat_weight
            loss_matrix_surface = criterion(output_surface, target_surface) * lat_weight
            
            loss_upper = torch.mean(loss_matrix_upper, dim=(0, 2, 3, 4)) * loss_weights['upper']
            loss_surface = torch.mean(loss_matrix_surface, dim=(0, 2, 3)) * loss_weights['surface']
            
            loss_nll = (torch.mean(loss_upper) * upper_surface_weights['upper'] +
                        torch.mean(loss_surface) * upper_surface_weights['surface'])

        loss_nll.backward()
        loss_kl = 0.
        with torch.no_grad():
            sig2 = self.prior_sig ** 2
            for (pname, p), p0, p_m, p_rho in zip(
                workhorse_net.named_parameters(),
                prior_net.parameters(), 
                self.m.parameters(),    
                self.rho.parameters()  
            ):
                s = F.softplus(p_rho)
                v = s ** 2
                d = p.numel()
                
                loss_kl += 0.5 * ((((p_m - p0) ** 2 + v) / sig2).sum() - (v / sig2).log().sum() - d)

                if p_m.grad is None: p_m.grad = torch.zeros_like(p_m)
                if p_rho.grad is None: p_rho.grad = torch.zeros_like(p_rho)
                p_m.grad.copy_(p.grad + beta * (p_m - p0) / sig2 / self.ND)

                g_s = p.grad * ((p - p_m) / s) + beta * (s / sig2 - 1.0 / s) / self.ND
                p_rho.grad.copy_(g_s * torch.sigmoid(p_rho))

        if dist.is_initialized() and self.training:
            for p_m, p_rho in zip(self.m.parameters(), self.rho.parameters()):

                if p_m.grad is not None:
                    dist.all_reduce(p_m.grad, op=dist.ReduceOp.SUM)
                    p_m.grad /= dist.get_world_size()
                if p_rho.grad is not None:
                    dist.all_reduce(p_rho.grad, op=dist.ReduceOp.SUM)
                    p_rho.grad /= dist.get_world_size()

        loss_total = loss_nll + beta * loss_kl / self.ND
        return loss_total.item(), loss_nll.item(), loss_kl.item()

def main(args):

    dist.init_process_group(backend='nccl', timeout=timedelta(seconds=7200000))
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    if global_rank == 0: 
        print("Initializing models...")
    
    base_net_cpu = PanguWeather(args.num_class, args.input_channels, args.input_channels_surface,
                          args.output_channels, args.output_channels_surface,
                          args.init_channels, depths=utils.parse_list(args.depths),
                          window_size=utils.parse_list(args.window_size),
                          patch_size=utils.parse_list(args.patch_size),
                          num_heads=utils.parse_list(args.num_heads),
                          drop_path_rate=args.drop_path_rate,
                          split_size=args.split_size, attn_drop=args.attn_drop,
                          influence_length=10, influence_length_surface=8
                          )

    workhorse_net = copy.deepcopy(base_net_cpu).to(device)
    prior_net = base_net_cpu
    state_dict = torch.load('/home/u2024110474/xiongxinlei/record/checkpoint/Pangu_2.pt', map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    prior_net.load_state_dict(new_state_dict)
    prior_net.to(device)
    net_init = copy.deepcopy(prior_net) 
    for p in prior_net.parameters():
        p.requires_grad_(False)

    if global_rank == 0: print("Loading data...")
    train_dataset = WeatherDataset40(patch_size=utils.parse_list(args.patch_size), is_train=True)
    train_sampler = DistributedSampler(train_dataset, rank=global_rank, shuffle=True)
    train_loader = DataLoader(train_dataset, args.batch_size, sampler=train_sampler, num_workers=args.num_workers, drop_last=True, pin_memory=True)

    test_dataset = WeatherDataset40test([2, 4, 4], steps = 1, is_train = False)
    sampler = DistributedSampler(test_dataset, rank = global_rank, shuffle =False)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle = False, sampler = sampler, num_workers = args.num_workers, drop_last = False, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    
    if global_rank == 0: print(f"Training dataset size: {len(train_dataset)}")

    if global_rank == 0: print("Setting up VI model and optimizer...")

    vi_model = VI_Model(net_init=net_init, ND=len(train_dataset), prior_sig=args.prior_sig).to(device)

    optimizer = torch.optim.AdamW(vi_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.epochs)
    scaler = GradScaler()

    criterion = nn.L1Loss(reduction='none')
    lat_weight = utils.cal_lat_weight().to(device, dtype=torch.float32)
    upper_weight = torch.tensor(1.0, device=device, dtype=torch.float32)
    surface_weight = torch.tensor(0.25, device=device, dtype=torch.float32)
    loss_weight_upper_train = torch.tensor(np.array([3.00, 0.60, 1.50, 0.77, 0.54])).to(device, dtype=torch.float32)
    loss_weight_surface_train = torch.tensor(np.array([3.00, 0.77, 0.66, 1.50])).to(device, dtype=torch.float32)
    
    topography = torch.from_numpy(np.load('../ERA5_examples/constant_masks/topography.npy').astype(np.float32))
    land_mask = torch.from_numpy(np.load('../ERA5_examples/constant_masks/land_mask.npy').astype(np.float32))
    soil_type = torch.from_numpy(np.load('../ERA5_examples/constant_masks/soil_type.npy').astype(np.float32))
    constant = torch.stack([(topography - topography.mean()) / topography.std(), land_mask, soil_type], dim=0).to(device=device)
    mean_upper = torch.load(('../ERA5_examples/statistic/upper_mean_40.pt'), weights_only=False).unsqueeze(-1).unsqueeze(-1).to(device=device)
    std_upper = torch.load(('../ERA5_examples/statistic/upper_std_40.pt'), weights_only=False).unsqueeze(-1).unsqueeze(-1).to(device=device)

    loss_fn_dict = {
        'criterion': criterion, 'scaler': scaler, 'lat_weight': lat_weight,
        'loss_weights': {'upper': loss_weight_upper_train, 'surface': loss_weight_surface_train},
        'upper_surface_weights': {'upper': upper_weight, 'surface': surface_weight}
    }

    if global_rank == 0: print("Starting training...")
    
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        vi_model.train()
        workhorse_net.train()

        for idx, (input_upper, input_surface, target_upper, target_surface) in enumerate(train_loader):

            input_upper = input_upper.to(device, non_blocking=True)
            input_surface = input_surface.to(device, non_blocking=True)
            target_upper = target_upper.to(device, non_blocking=True)
            target_surface = target_surface.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            workhorse_net.zero_grad(set_to_none=True) 
            
            total_loss, nll_loss, kl_loss = vi_model(
                workhorse_net, prior_net,
                inputs=(input_upper, input_surface, constant),
                targets=(target_upper, target_surface),
                loss_fn_dict=loss_fn_dict,
                beta = 1e-4) 
            
            optimizer.step()
            scheduler.step()
            
            if (idx + 1) % args.log_interval == 0 and global_rank == 0:
                current_lr = optimizer.param_groups[0]['lr']
                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f"[{current_time}] "
                    f"Epoch {epoch+1}/{args.epochs}, Batch {idx+1}/{len(train_loader)}, "
                    f"LR: {current_lr:.8f}, "
                    f"Total Loss: {total_loss:.4f}, NLL: {nll_loss:.4f}, KL: {kl_loss:.4f}")
                
        if global_rank == 0:
            torch.save(vi_model.state_dict(), f"../record/checkpoint/pangu_8_24_vi_epoch{epoch+1}.pt")

        vi_model.eval()
        workhorse_net.eval()
        local_rmse_z500 = []
        local_spread_z500 = []
        
        with torch.inference_mode():
            for idx, (input_upper, input_surface, target_upper, target_surface) in enumerate(test_loader):
                input_upper = input_upper.to(device)
                input_surface = input_surface.to(device)
                target_upper = target_upper.to(device)
                target_surface = target_surface.to(device)
                
                ensemble_members_upper = []
                for i in range(8):
                    m_params = vi_model.m.parameters()
                    rho_params = vi_model.rho.parameters()
                    for p, p_m, p_rho in zip(workhorse_net.parameters(), m_params, rho_params):
                        eps = torch.randn_like(p)
                        s = F.softplus(p_rho)
                        p.copy_(p_m + s * eps)

                    output_upper, out_surface = workhorse_net(input_upper, input_surface, constant, 13, 721, 1440)
                    output_upper = output_upper * std_upper + mean_upper
                    ensemble_members_upper.append(output_upper)
                
                full_ensemble_upper = torch.stack(ensemble_members_upper, dim=1)
                output_upper_mean = full_ensemble_upper.mean(dim=1)

                local_rmse_z500.append(cal_mse(output_upper_mean[:, 0, 7, :, :], target_upper[:, 0, 7, :, :], lat_weight))
                local_spread_z500.append(cal_spread(full_ensemble_upper[:, :, 0, 7, :, :], lat_weight))

        local_rmse_z500 = torch.stack(local_rmse_z500)
        local_spread_z500 = torch.stack(local_spread_z500)
        world_size = dist.get_world_size()
        gathered_rmse_z500 = [torch.zeros_like(local_rmse_z500) for _ in range(world_size)]
        gathered_spread_z500 = [torch.zeros_like(local_spread_z500) for _ in range(world_size)]
        
        dist.all_gather(gathered_rmse_z500, local_rmse_z500)
        dist.all_gather(gathered_spread_z500, local_spread_z500)
        
        if int(local_rank) == 0:
            mean_rmse_z500 = torch.sqrt(torch.mean(torch.stack(gathered_rmse_z500)))
            mean_spread_z500 = torch.sqrt(torch.mean(torch.stack(gathered_spread_z500)))
            print('Epoch: {}/{}'.format(epoch + 1, args.epochs))
            print('rmse_z500: {:.2f}'.format(mean_rmse_z500))
            print('spread_z500: {:.2f}'.format(mean_spread_z500))
            torch.cuda.empty_cache()

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pangu-VI")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default = 3e-4)
    parser.add_argument('--epochs', type=int, default = 5)
    parser.add_argument('--weight_decay', type=float, default = 0.1)
    parser.add_argument('--depths', type=str, default='8_24')
    parser.add_argument('--patch_size', type=str, default='2_4_4')
    parser.add_argument('--window_size', type=str, default='2_6_12')
    parser.add_argument('--num_heads', type=str, default='6_12')
    parser.add_argument('--drop_path_rate', type=float, default= 0.0)
    parser.add_argument('--attn_drop', type=float, default= 0.0)
    parser.add_argument("--log_interval", type=int, default= 500)
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default = 4)
    parser.add_argument('--input_channels', type=int, default= 10)
    parser.add_argument('--input_channels_surface', type=int, default=11)
    parser.add_argument('--output_channels', type=int, default=5)
    parser.add_argument('--output_channels_surface', type=int, default=4)
    parser.add_argument('--init_channels', type=int, default=192)
    parser.add_argument('--input_shape', type=str, default="13_721_1440")
    parser.add_argument('--split_size', type=int, default=1)
    parser.add_argument('--influence_length', type=int, default=10)
    parser.add_argument('--influence_length_surface', type=int, default=8)
    parser.add_argument('--world_size', type=int, default=8)
    parser.add_argument('--max_norm', type=float, default=5.0)
    parser.add_argument('--hour', type=int, default=4)
    parser.add_argument('--prior_sig', type=float, default=1)
    parser.add_argument('--eval_freq', type=int, default=1)
    args = parser.parse_args()
    main(args)