import torch
import torch.nn as nn
import utils
import argparse
import numpy as np
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from utils import DropPath, trunc_normal_

class PatchEmbed(nn.Module):
    def __init__(self, patch_size, input_channels, input_channels_surface, output_channels, input_shape):
        super().__init__()
        self.patch_size = patch_size
        self._input_channels = input_channels
        self._output_channels = output_channels
        self.proj = nn.Conv1d(input_channels * patch_size[0] * patch_size[1] * patch_size[2], output_channels, kernel_size=1, stride=1, groups=1)
        self.proj_surface = nn.Conv1d(input_channels_surface*patch_size[1]*patch_size[2], output_channels, kernel_size=1, stride=1, groups=1)
        self.land_mask_table = nn.Parameter(torch.zeros((2)))
        self.soil_type_table = nn.Parameter(torch.zeros((8)))
        self.OZ = None
        self.OH = None
        self.OW = None

    def forward(self, x, x_surface, x_constant):
        B, C, Z, H, W = x.shape     # B*5*13*721*1440
        x_z = x_constant[0:1, :, :]     # 3*721*1440
        x_land_mask = F.one_hot(x_constant[1:2, :, :].to(torch.int64)) * self.land_mask_table
        x_land_mask = x_land_mask.sum(dim=-1)
        x_soil_type = F.one_hot(x_constant[2:, :, :].to(torch.int64)) * self.soil_type_table
        x_soil_type = x_soil_type.sum(dim=-1)
        x_surface_constant = torch.cat([x_z, x_land_mask, x_soil_type], dim=0)
        x_surface_constant = x_surface_constant.view(1, x_surface_constant.shape[0], x_surface_constant.shape[1], x_surface_constant.shape[2]).repeat(B, 1, 1, 1)
        B, _, Hc, Wc = x_surface_constant.shape
        if Wc % self.patch_size[2] != 0:
            x_surface_constant = F.pad(x_surface_constant, (0, self.patch_size[2] - Wc % self.patch_size[2]))

        if Hc % self.patch_size[1] != 0:
            x_surface_constant = F.pad(x_surface_constant, (0, 0, 0, self.patch_size[1] - Hc % self.patch_size[1]))
        x_surface = torch.cat([x_surface, x_surface_constant], dim=1)
        B, C_surface, H, W = x_surface.shape
        x = x.view(B, C, Z //self.patch_size[0], self.patch_size[0], H // self.patch_size[1], self.patch_size[1], W // self.patch_size[2], self.patch_size[2])
        x_surface = x_surface.view(B, C_surface, H // self.patch_size[1], self.patch_size[1], W // self.patch_size[2], self.patch_size[2])
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x_surface = x_surface.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * self.patch_size[0] * self.patch_size[1] * self.patch_size[2], (Z // self.patch_size[0]) * (H // self.patch_size[1]) * (W // self.patch_size[2]))
        x_surface = x_surface.view(B, C_surface * self.patch_size[1] * self.patch_size[2], (H // self.patch_size[1]) * (W // self.patch_size[2]))
        # forward function
        # padding
        x = self.proj(x) # B C Wh Ww
        x_surface = self.proj_surface(x_surface)
        B, C_out, L_3d = x.shape
        x = x.view(B, C_out, Z // self.patch_size[0], H // self.patch_size[1], W // self.patch_size[2])
        x_surface = x_surface.view(B, C_out, 1, H // self.patch_size[1], W // self.patch_size[2])
        x = torch.cat([x_surface, x], dim=2)
        self.OZ, self.OH, self.OW = 1 + Z // self.patch_size[0], H // self.patch_size[1], W // self.patch_size[2]
        x = x.view(B, C_out, self.OZ * self.OH * self.OW)
        x = x.permute(0, 2, 1)
        return x

class PatchRecover(nn.Module):
    def __init__(self, patch_size, input_channels, output_channels, output_channels_surface, input_shape):
        super().__init__()
        self.patch_size = patch_size
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._output_channels_surface = output_channels_surface
        self.input_shape = input_shape
        self.proj = nn.Conv1d(input_channels, output_channels * patch_size[0] * patch_size[1] * patch_size[2], kernel_size=1, stride=1, groups=1)
        self.proj_surface = nn.Conv1d(input_channels, output_channels_surface * patch_size[1] * patch_size[2], kernel_size=1, stride=1, groups=1)
        self.Z = None
        self.H = None
        self.W = None
        self.IZ = None
        self.IH = None
        self.IW = None

    def forward(self, x):
        Z, H, W, IZ, IH, IW = self.Z, self.H, self.W, self.IZ, self.IH, self.IW
        B, L, C = x.shape
        assert L == Z * H * W
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B, C, Z, H, W)
        x_3d = x[:, :, 1:, :, :].contiguous().view(B, C, (Z - 1) * H * W)
        x_surface = x[:, :, 0, :, :].contiguous().view(B, C, H * W)
        x_3d = self.proj(x_3d)  # B C Wh Ww
        x_surface = self.proj_surface(x_surface)
        B, C_all, L_all_3d = x_3d.shape

        x_3d = x_3d.view(B, self._output_channels, self.patch_size[0], self.patch_size[1], self.patch_size[2], Z - 1, H, W).permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x_surface = x_surface.view(B, self._output_channels_surface, self.patch_size[1], self.patch_size[2], H, W).permute(0, 1, 4, 2, 5, 3).contiguous()
        x_3d = x_3d.view(B, self._output_channels, (Z - 1) * self.patch_size[0], H * self.patch_size[1], W * self.patch_size[2])
        x_surface = x_surface.view(B, self._output_channels_surface, H * self.patch_size[1], W * self.patch_size[2])
        x_3d = x_3d[:, :, :IZ, :IH, :IW]
        x_surface = x_surface[:, :, :IH, :IW]
        return x_3d, x_surface

def window_reverse(windows, window_size, Z, H, W):
    B_, tW, Nw, C = windows.shape
    B = int(B_ / (W // window_size[2]))
    x = windows.view(B, W // window_size[2], Z // window_size[0], H // window_size[1], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 2, 4, 3, 5, 1, 6, 7).contiguous().view(B, Z, H, W, -1)
    return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act_layer = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_layer(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., input_shape=None):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.input_shape = input_shape
        self.type_of_windows = (input_shape[0] // window_size[0]) * (input_shape[1] // window_size[1])

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[2] - 1) * window_size[1] * window_size[1] * window_size[0] * window_size[0], self.type_of_windows, num_heads))
        coords_zi = torch.arange(self.window_size[0])
        coords_zj = -torch.arange(self.window_size[0]) * self.window_size[0]
        coords_hi = torch.arange(self.window_size[1])
        coords_hj = -torch.arange(self.window_size[1]) * self.window_size[1]
        coords_w = torch.arange(self.window_size[2])
        coords_1 = torch.stack(torch.meshgrid([coords_zi, coords_hi, coords_w]))    # 2, Wh, Ww
        coords_2 = torch.stack(torch.meshgrid([coords_zj, coords_hj, coords_w]))
        coords_flatten_1 = torch.flatten(coords_1, start_dim=1)     # 2, Wh * Ww
        coords_flatten_2 = torch.flatten(coords_2, start_dim=1)
        relative_coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]   # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()     # Wh*Ww, Wh*Ww, 2
        # shift to start from 0
        relative_coords[:, :, 2] += window_size[2] - 1
        relative_coords[:, :, 1] *= 2 * window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * window_size[2] - 1) * window_size[1] * window_size[1]
        relative_position_index = relative_coords.sum(-1)   # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, tW, N, C = x.shape
        qkv = self.qkv(x).view(B_, tW, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2], self.window_size[0] * self.window_size[1] * self.window_size[2], self.type_of_windows, self.num_heads)     # Wh*Ww, Wh*Ww, nH
        relative_position_bias = relative_position_bias.permute(2, 3, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, tW, self.num_heads, N, N) + mask.unsqueeze(2).unsqueeze(0)
            attn = attn.view(B_, tW, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(2, 3).reshape(B_, tW, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, split_size=1, window_size=(7,7,3), shift_size=(0,0,0), mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, input_shape=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        input_shape = [utils.calculate_depart_shape(input_shape[0], window_size[0]), utils.calculate_depart_shape(input_shape[1], window_size[1]), utils.calculate_depart_shape(input_shape[2], window_size[2])]
        self.input_shape = input_shape
        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, input_shape=input_shape)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.split_size = split_size

        self.H = None
        self.W = None
        self.Z = None

    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        Z, H, W = self.Z, self.H, self.W
        assert  L == Z * H * W, "input feature has wrong size"
        shortcut = x
        x = x.view(B, Z, H, W, C)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_z1 = 0
        pad_z2 = (self.window_size[0] - Z % self.window_size[0]) % self.window_size[0]
        pad_r = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        pad_b = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_z1, pad_z2))
        _, Zp, Hp, Wp, _ = x.shape
        # cyclic shift
        if self.shift_size[0] > 0 or self.shift_size[1] > 0 or self.shift_size[2] > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = utils.window_partition(shifted_x, self.window_size)
        Bw, tW, w1, w2, w3, C = x_windows.shape
        x_windows = x_windows.view(Bw, tW, self.window_size[0] * self.window_size[1] * self.window_size[2], C)

        x_windows_split = torch.chunk(x_windows, chunks=self.split_size, dim=0)
        if attn_mask is not None:
            attn_mask_split = torch.chunk(attn_mask, chunks=self.split_size, dim=0)
        else:
            attn_mask_split = [None] * self.split_size
        i = 0
        attn_windows = []
        for x_window_slice in x_windows_split:
            attn_windows_slice = checkpoint(self.attn, x_window_slice, attn_mask_split[i], use_reentrant=False)
            # attn_windows_slice = self.attn(x_window_slice, attn_mask_split[i])
            attn_windows.append(attn_windows_slice)
            i += 1
        attn_windows = torch.cat(attn_windows, dim=0)

        # merge windows
        shifted_x = window_reverse(attn_windows, self.window_size, Zp, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size[0] > 0 or self.shift_size[1] > 0 or self.shift_size[2] > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0 or pad_z2 > 0:
            x = x[:, :Z, :H, :W, :].contiguous()
        x = x.view(B, Z * H * W, C)

        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        self.Z = None
        self.H = None
        self.W = None

    def forward(self, x):
        Z, H, W = self.Z, self.H, self.W
        B, L, C = x.shape
        assert L == Z * H * W, "input feature has wrong size"

        x = x.view(B, Z, H, W, C)
        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        B, Z, H, W, C = x.shape
        x = x.view(B, Z, H // 2, 2, W // 2, 2, C).permute(0, 1, 2, 4, 3, 5, 6).contiguous().view(B, Z * (H // 2) * (W //2), 4 * C)

        x = self.norm(x)
        x = self.reduction(x)
        return x

class PatchUpsample(nn.Module):
    def __init__(self, input_dim, output_dim, norm_layer=nn.LayerNorm, up_dims_mode=1):
        super().__init__()
        compress_ratio_list = [2, 4, 8]
        compress_ratio = compress_ratio_list[up_dims_mode]
        self.preprocess = nn.Linear(input_dim, compress_ratio * output_dim, bias=False)
        self.mixup = nn.Linear(output_dim, output_dim, bias=False)
        self.norm = norm_layer(output_dim)
        self.Z = None
        self.H = None
        self.W = None
        self.OZ = None
        self.OH = None
        self.OW = None

    def forward(self, x):
        Z, H, W, OZ, OH, OW = self.Z, self.H, self.W, self.OZ, self.OH, self.OW
        x = self.preprocess(x)
        B, L, C = x.shape
        assert L == Z * H * W, "input feature has wrong size"
        x = x.view(B, Z, H, W, 2, 2, C // 4).permute(0, 1, 2, 4, 3, 5, 6).contiguous().view(B, Z, H*2, W*2, C//4)[:, :, :OH, :OW, :]
        x = x.contiguous().view(B, -1, C//4)
        x = self.norm(x)
        x = self.mixup(x)
        return x

class BaseBlock(nn.Module):
    def __init__(self, dim, depth, num_heads, split_size=1, window_size=[7,7,7],
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, input_shape=None):
        super().__init__()
        self.window_size = window_size
        if window_size[0] == 7:
            self.shift_size = [0, window_size[1] // 2, window_size[2] // 2]
        else:
            self.shift_size = [window_size[0] // 2, window_size[1] // 2, window_size[2] // 2]
        self.depth = depth

        self.blocks = nn.ModuleList()
        for i in range(depth):
            blk = SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                split_size=split_size,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=None,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                input_shape=input_shape
            )
            self.blocks.append(blk)

    def forward(self, x, Z, H, W):
        Hp = int(np.ceil(H / self.window_size[1])) * self.window_size[1]
        Wp = int(np.ceil(W / self.window_size[2])) * self.window_size[2]
        Zp = int(np.ceil(Z / self.window_size[0])) * self.window_size[0]
        img_mask = torch.zeros((1, Zp, Hp, Wp, 1), device=x.device)     # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        z_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        cnt = 0
        for z in z_slices:
            for h in h_slices:
                img_mask[:, z, h, :, :] = cnt
                cnt += 1

        mask_windows = utils.window_partition(img_mask, self.window_size)
        Bw, tW, w1, w2, w3, C = mask_windows.shape
        mask_windows = mask_windows.view(Bw, tW, self.window_size[0] * self.window_size[1] * self.window_size[2])
        attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(3)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.Z, blk.H, blk.W = Z, H, W
            x = checkpoint(blk, x, attn_mask, use_reentrant=False)
            # x = blk(x, attn_mask)
        return x, Z, H, W

class EncoderBlock(nn.Module):
    def __init__(self, dim, depth, num_heads, split_size=1, window_size=[7,7,7],
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, input_shape=None):
        super().__init__()
        self._base_op = BaseBlock(dim=dim, depth=depth, num_heads=num_heads,
                                  split_size=split_size, window_size=window_size,
                                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                                  attn_drop=attn_drop, drop_path=drop_path,
                                  norm_layer=norm_layer, input_shape=input_shape)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, Z, H, W):
        x, Z, H, W = self._base_op(x, Z, H, W)
        if self.downsample is not None:
            self.downsample.Z, self.downsample.H, self.downsample.W = Z, H, W
            x_down = checkpoint(self.downsample, x, use_reentrant=False)
            # x_down = self.downsample(x)
            DZ, DH, DW = Z, (H + 1) // 2, (W + 1) // 2
            return x_down, x, Z, H, W, DZ, DH, DW
        else:
            return None, x, Z, H, W, Z, H, W

class DecoderBlock(nn.Module):
    def __init__(self, input_dim, output_dim, depth, num_heads, split_size=1, window_size=[7,7,7],
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, upsample=None, up_dims_mode=1, input_shape=None):
        super().__init__()
        if upsample is not None:
            self.upsample = upsample(input_dim=input_dim, output_dim=output_dim, norm_layer=norm_layer, up_dims_mode=up_dims_mode)
        else:
            self.upsample = None
        self._base_op = BaseBlock(output_dim, depth=depth, num_heads=num_heads, split_size=split_size, window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, input_shape=input_shape)

    def forward(self, x, Z, H, W, OZ=None, OH=None, OW=None):
        if self.upsample is not None:
            self.upsample.Z, self.upsample.H, self.upsample.W, self.upsample.OZ, self.upsample.OH,  self.upsample.OW= Z, H, W, OZ, OH, OW
            x = checkpoint(self.upsample, x, use_reentrant=False)
            # x = self.upsample(x)
            Z, H, W = OZ, OH, OW
        x, Z, H, W = self._base_op(x, Z, H, W)
        return x, Z, H, W


class PanguWeather(nn.Module):
    def __init__(self, num_class, input_channels, input_channels_surface, output_channels, output_channels_surface,
                 init_channels, input_shape=[13, 721, 1440], patch_size=[2,4,4],
                 window_size=[2,6,12], depths=[8,24], drop_path_rate=0.2,
                 num_heads=[6,12], split_size=1, attn_drop=0., influence_length=5,
                 influence_length_surface=4):
        super().__init__()
        self._input_shape = input_shape
        self.padding_shape = [utils.calculate_depart_shape(self._input_shape[0], patch_size[0]), utils.calculate_depart_shape(self._input_shape[1], patch_size[1]), utils.calculate_depart_shape(self._input_shape[2], patch_size[2])]
        self._stem = PatchEmbed(patch_size=patch_size, input_channels=input_channels, input_channels_surface=input_channels_surface, output_channels=init_channels, input_shape=self.padding_shape)
        self.influence_length = influence_length
        self.influence_length_surface = influence_length_surface
        self._embeded_shape = [utils.calculate_depart_shape(self._input_shape[0], patch_size[0]) // patch_size[0], utils.calculate_depart_shape(self._input_shape[1], patch_size[1]) // patch_size[1], utils.calculate_depart_shape(self._input_shape[2], patch_size[2]) // patch_size[2]]
        self.num_layers = len(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self._encoder_list = nn.ModuleList()
        self._decoder_list = nn.ModuleList()
        curr_shape = self._embeded_shape
        for i_layer in range(self.num_layers):
            curr_channels = int(init_channels * 2 ** i_layer)
            encoder_layer = EncoderBlock(dim=curr_channels, depth=int(depths[i_layer]),
                                         num_heads=num_heads[i_layer], split_size=split_size,
                                         window_size=window_size, mlp_ratio=4, qkv_bias=True,
                                         drop=0., attn_drop=attn_drop, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                         norm_layer=nn.LayerNorm, downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                         input_shape=curr_shape)
            if i_layer < self.num_layers - 2:
                c_in = 4 * curr_channels
            elif i_layer == self.num_layers - 2:
                c_in = 2 * curr_channels
            else:
                c_in = curr_channels
            decoder_layer = DecoderBlock(input_dim=c_in, output_dim=curr_channels,
                                         depth=int(depths[i_layer]), num_heads=num_heads[i_layer],
                                         split_size=split_size, window_size=window_size,
                                         mlp_ratio=4, qkv_bias=True, drop=0., attn_drop=attn_drop,
                                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                         norm_layer=nn.LayerNorm, upsample=PatchUpsample if (i_layer < self.num_layers - 1) else None,
                                         input_shape=curr_shape)
            self._encoder_list.append(encoder_layer)
            self._decoder_list.append(decoder_layer)
            curr_shape = [curr_shape[0], (curr_shape[1] + 1) // 2, (curr_shape[2] + 1) // 2]
        curr_channels = init_channels * 2
        self._output_layer = PatchRecover(patch_size=patch_size, input_channels=curr_channels, output_channels=output_channels, output_channels_surface=output_channels_surface, input_shape=self._embeded_shape)
    def forward(self, x, x_surface, x_constant, Z, H, W):
        IZ, IH, IW = Z, H, W
        x = self._stem(x, x_surface, x_constant)
        Z, H, W = self._stem.OZ, self._stem.OH, self._stem.OW
        shapes_in = []
        states = []
        i = 0
        for op in self._encoder_list:
            x, x_out, OZ, OH, OW, Z, H, W = op(x, Z, H, W)
            if x is None:
                x = x_out
            shapes_in.append((OZ, OH, OW))
            states.append(x_out)
            i += 1
        x = states[-1]
        i = 0
        for i in range(self.num_layers):
            OZ, OH, OW = shapes_in[self.num_layers - i - 1]
            x, Z, H, W = self._decoder_list[self.num_layers - i - 1](x, Z, H, W, OZ, OH, OW)
            if i > 0:
                x = torch.cat([states[self.num_layers - i - 1], x], dim=2)
            Z, H, W = OZ, OH, OW
        self._output_layer.Z, self._output_layer.H, self._output_layer.W, self._output_layer.IZ, self._output_layer.IH, self._output_layer.IW = Z, H, W, IZ, IH, IW
        x, x_surface = self._output_layer(x)
        return x, x_surface


class PanguPerturbWrapper(nn.Module):
    def __init__(self, input_c_upper, input_c_surface, 
                 output_c_upper, output_c_surface):
        super().__init__()
        self.core_model = PanguWeather(
            num_class=0, 
            input_channels=input_c_upper,
            input_channels_surface=input_c_surface,
            output_channels=10,
            output_channels_surface=8,
            init_channels=192,           
            depths=[2,6],        
            num_heads=[6,12],   
            patch_size=[2, 4, 4],      
            window_size=[2, 6, 12],     
            input_shape=[13, 721, 1440], 
            drop_path_rate=0.0
        )
    def forward(self, x, x_surface, x_constant, Z, H, W):
        def run_core(x_in, x_surf_in, x_const_in):
            return self.core_model(x_in, x_surf_in, x_const_in, Z, H, W)
        if self.training:
            if not x.requires_grad:
                x.requires_grad_(True)
            out_u, out_s = checkpoint(run_core, x, x_surface, x_constant, use_reentrant=False)
        else:
            out_u, out_s = run_core(x, x_surface, x_constant)

        mu_u, raw_sigma_u = torch.chunk(out_u, 2, dim=1)
        mu_s, raw_sigma_s = torch.chunk(out_s, 2, dim=1)
        
        sigma_u = F.softplus(raw_sigma_u - 3.0) + 1e-6
        sigma_s = F.softplus(raw_sigma_s - 3.0) + 1e-6
        
        return mu_u, sigma_u, mu_s, sigma_s