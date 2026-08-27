import torch
import utils
import os
import argparse
from network import PanguWeather
import torch.optim as optim
from datetime import timedelta
import warnings
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from data_idx import WeatherDataset40, WeatherDataset40test
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from network import PanguWeather
import torch.optim as optim
import torch.nn as nn
from utils import cal_mae, cal_rmse, clip_grad_norm_, cal_mse
import numpy as np
import time
warnings.filterwarnings("ignore")

def main(args):
    torch.cuda.empty_cache()
    dist.init_process_group(backend='nccl', timeout=timedelta(seconds=7200000))
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    model = PanguWeather(args.num_class, args.input_channels, args.input_channels_surface,
                         args.output_channels, args.output_channels_surface,
                         args.init_channels, depths=utils.parse_list(args.depths),
                         window_size=utils.parse_list(args.window_size),
                         patch_size=utils.parse_list(args.patch_size),
                         num_heads=utils.parse_list(args.num_heads),
                         drop_path_rate=args.drop_path_rate,
                         split_size=args.split_size, attn_drop=args.attn_drop,
                         influence_length=10, influence_length_surface=8
                         ).to(device, dtype=torch.float32)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters：{total_params}")
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    checkpoint = torch.load('/home/u2024110474/xiongxinlei/NMI/train_scripts/Pangu_muon_244_6_18_2.pt', map_location='cpu') 
    model.load_state_dict(checkpoint, strict= True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95))
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

    train_dataset = WeatherDataset40([2, 4, 4], is_train = True)
    train_sampler = DistributedSampler(train_dataset, rank = global_rank, shuffle = True)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle = False, sampler = train_sampler, num_workers = args.num_workers, drop_last = True, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    print(len(train_loader))

    test_dataset = WeatherDataset40test([2, 4, 4], steps = 1, is_train = False)
    test_sampler = DistributedSampler(test_dataset, rank = global_rank, shuffle =False)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle = False, sampler = test_sampler, num_workers = args.num_workers, drop_last = False, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    print(len(test_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*args.epochs)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        start_epoch_time = time.time()
        model.train()
        for idx, (input_upper, input_surface, target_upper, target_surface) in enumerate(train_loader):

            B, V, Z, H, W = target_upper.shape
            input_upper, input_surface, target_upper, target_surface= input_upper.to(device), input_surface.to(device), target_upper.to(device), target_surface.to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(dtype=torch.bfloat16):
                output_upper, output_surface = model(input_upper, input_surface, constant, Z, H, W)
            loss_matrix_upper = criterion(output_upper, target_upper) * lat_weight
            loss_matrix_surface = criterion(output_surface, target_surface) * lat_weight

            loss_upper = torch.mean(loss_matrix_upper, dim=(0, 2, 3, 4)) * loss_weight_upper_train.to(loss_matrix_upper.device)
            loss_surface = torch.mean(loss_matrix_surface, dim=(0, 2, 3)) * loss_weight_surface_train.to(loss_matrix_surface.device)
            loss_all = torch.mean(loss_upper) * upper_weight + torch.mean(loss_surface) * surface_weight

            loss_all.backward()
            gradient_norm = clip_grad_norm_(model.module.parameters(), args.max_norm, clip_grad=False)
            optimizer.step()
            scheduler.step()

            if (idx + 1) % args.log_interval == 0:
                losses_tensor = torch.tensor([loss_all.item()]).to(device)
                dist.all_reduce(losses_tensor)
                mean_losses = losses_tensor / dist.get_world_size()
                
                if int(local_rank) == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    mean_total_loss = mean_losses.item()
                    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print(f"Epoch {epoch + 1}, Batch {idx + 1}," f"Total Loss: {mean_total_loss:.6f}," f" Grad Norm: {gradient_norm:.4f}, Time: {current_time}, "f"LR: {current_lr:.8f}")

        if int(local_rank) == 0:
            print(f"Epoch {epoch + 1} Training Time: {time.time() - start_epoch_time:.4f} seconds")
            torch.save(model.state_dict(), f"../record/checkpoint/Pangu_6_18_{epoch+3}.pt")
        
    local_mse_z500=[]
    model.eval()
    with torch.inference_mode():
        for idx, (input_upper, input_surface, target_upper, target_surface) in enumerate(test_loader):
            _, _, Z, H, W = target_upper.shape
            input_upper, input_surface, target_upper, target_surface = input_upper.to(device), input_surface.to(device),target_upper.to(device), target_surface.to(device)
            output_upper, output_surface = model(input_upper, input_surface, constant, Z, H, W)
            output_upper =  output_upper * std_upper + mean_upper
            local_mse_z500.append(cal_mse(output_upper[:, 0, 7, :, :], target_upper[:, 0, 7, :, :], lat_weight))
            print(cal_mse(output_upper[:, 0, 7, :, :], target_upper[:, 0, 7, :, :], lat_weight))

    local_mse_z500 = torch.stack(local_mse_z500)
    world_size = dist.get_world_size()
    gathered_mse_z500 = [torch.zeros_like(local_mse_z500) for _ in range(world_size)]
    dist.all_gather(gathered_mse_z500, local_mse_z500)
    if int(local_rank) == 0:
        mean_mse_z500 = torch.mean(torch.stack(gathered_mse_z500))
        mean_rmse_z500 = torch.sqrt(mean_mse_z500)
        print('rmse_z500: {:.2f}'.format(mean_rmse_z500))
        torch.cuda.empty_cache()

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("PanguWeather")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default = 8e-5)
    parser.add_argument('--epochs', type=int, default = 2)
    parser.add_argument('--weight_decay', type=float, default = 0.1)
    parser.add_argument('--depths', type=str, default='6_18')
    parser.add_argument('--patch_size', type=str, default='2_4_4')
    parser.add_argument('--window_size', type=str, default='2_6_12')
    parser.add_argument('--num_heads', type=str, default='6_12')
    parser.add_argument('--drop_path_rate', type=float, default= 0.2)
    parser.add_argument('--attn_drop', type=float, default= 0.0)
    parser.add_argument("--log_interval", type=int, default= 1)
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
    args = parser.parse_args()
    main(args)