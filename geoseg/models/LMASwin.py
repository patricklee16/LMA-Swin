import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn import Module, Conv2d, Parameter, Softmax
from geoseg.models.swin import SwinTransformer
from geoseg.models.MIRNet import *


class MaxPoolLayer(nn.Sequential):
    def __init__(self, kernel_size=3, dilation=1, stride=1):
        super(MaxPoolLayer, self).__init__(
            nn.MaxPool2d(kernel_size=kernel_size, dilation=dilation, stride=stride,
                         padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class AvgPoolLayer(nn.Sequential):
    def __init__(self, kernel_size=3, stride=1):
        super(AvgPoolLayer, self).__init__(
            nn.AvgPool2d(kernel_size=kernel_size, stride=stride,
                         padding=(kernel_size-1)//2)
        )


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class TransposeConvBNReLu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, norm_layer=nn.BatchNorm2d):
        super(TransposeConvBNReLu, self).__init__(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            norm_layer(out_channels),
            nn.ReLU()
        )


class TransposeConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, norm_layer=nn.BatchNorm2d):
        super(TransposeConvBN, self).__init__(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            norm_layer(out_channels)
        )


class TransposeConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(TransposeConv, self).__init__(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        )


class PyramidPool(nn.Sequential):
    def __init__(self, in_channels, out_channels, pool_size=1, norm_layer=nn.BatchNorm2d):
        super(PyramidPool, self).__init__(
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # print(self.relative_position_bias_table.shape)
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        # print(coords_h, coords_w)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        # print(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # print(coords_flatten)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        # print(relative_coords[0,7,:])
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # print(relative_coords[:, :, 0], relative_coords[:, :, 1])
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        # print(B_,N,C)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        # print(attn.shape)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # print(relative_position_bias.unsqueeze(0))
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x



class SwinTransformer(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        pretrain_img_size (int): Input image size for training the pretrained models,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=128,
                 depths=[2, 2, 18, 2],
                 num_heads=[4, 8, 16, 32],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.3,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        self.apply(self._init_weights)

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)
        # print('patch_embed', x.size())

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
                # print('layer{} out size {}'.format(i, out.size()))

        return tuple(outs)

    def train(self, mode=True):
        """Convert the models into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()


def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))




class DownConnection(nn.Module):
    def __init__(self, inplanes, planes, stride=2):
        super(DownConnection, self).__init__()
        self.convbn1 = ConvBN(inplanes, planes, kernel_size=3, stride=1)
        self.convbn2 = ConvBN(planes, planes, kernel_size=3, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = ConvBN(inplanes, planes, stride=stride)

    def forward(self, x):
        residual = x
        x = self.convbn1(x)
        x = self.relu(x)
        x = self.convbn2(x)
        x = x + self.downsample(residual)
        x = self.relu(x)

        return x

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                # nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))  # what is f(x)
        return torch.cat(out, 1)


class FM(nn.Module):
    def __init__(self, encoder_channels=(96, 192, 384, 768), atrous_rates=(6, 12), num_classes=6, dropout=0.05,
                 bias=False):
        super(FM, self).__init__()

        self.E_block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False))

        self.E_block2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False))

        self.E_block3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False))

        self.E_block4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False))

        self.E_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False))

        self._block1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2))

        self._block2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2))

        self._block3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2))

        self._block4 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2))

        self._block5 = nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2))

        self._block6 = nn.Sequential(
            nn.Conv2d(46, 23, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(23, 23, 3, stride=1, padding=1),
            nn.UpsamplingBilinear2d(scale_factor=2))

        self._block7 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 3, 3, stride=1, padding=1))

        self.embedding_dim = 3
        self.conv0 = nn.Conv2d(3, self.embedding_dim, 3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(256 * 3, 256, 3, stride=1, padding=1)
        self.conv1_1 = nn.Conv2d(384, 256, 3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(384, 256, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128 * 3, 128, 3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(192, 128, 3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(192, 128, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64 * 3, 64, 3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(96, 64, 3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(96, 64, 3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(48, 24, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(96, 48, 3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(48, 24, 3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(24, 12, 3, stride=1, padding=1)
        self.conv4_4 = nn.Conv2d(12, 12, 3, stride=1, padding=1)
        self.in_chans = 3
        self.maxpool = nn.MaxPool2d(2, stride=4, return_indices=False, ceil_mode=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.ReLU = nn.ReLU(inplace=True)
        self.IN_1 = nn.InstanceNorm2d(48, affine=False)
        self.IN_2 = nn.InstanceNorm2d(96, affine=False)
        self.IN_3 = nn.InstanceNorm2d(192, affine=False)
        self.PPM1 = PPM(32, 8, bins=(1, 2, 3, 4))
        self.PPM2 = PPM(64, 16, bins=(1, 2, 3, 4))
        self.PPM3 = PPM(128, 32, bins=(1, 2, 3, 4))
        self.PPM4 = PPM(256, 64, bins=(1, 2, 3, 4))

        self.MSRB1 = MSRB(256, 3, 1, 2, bias)
        self.MSRB2 = MSRB(128, 3, 1, 2, bias)
        self.MSRB3 = MSRB(64, 3, 1, 2, bias)
        self.MSRB4 = MSRB(32, 3, 1, 2, bias)

        self.swin_1 = SwinTransformer(pretrain_img_size=224,
                                      patch_size=2,
                                      in_chans=3,
                                      embed_dim=96,
                                      depths=[2, 2, 2],
                                      num_heads=[3, 6, 12],
                                      window_size=7,
                                      mlp_ratio=4.,
                                      qkv_bias=True,
                                      qk_scale=None,
                                      drop_rate=0.,
                                      attn_drop_rate=0.,
                                      drop_path_rate=0.2,
                                      norm_layer=nn.LayerNorm,
                                      ape=False,
                                      patch_norm=True,
                                      out_indices=(0, 1, 2),
                                      frozen_stages=-1,
                                      use_checkpoint=False)

        rate_1, rate_2 = tuple(atrous_rates)
        self.lf = nn.Sequential(
                                 SeparableConvBNReLU(encoder_channels[-3], encoder_channels[-4], dilation=rate_2),
                                 nn.UpsamplingNearest2d(scale_factor=2))

    def forward(self, x):
        swin_in = x  # 96,192,384,768
        swin_out_1 = self.swin_1(swin_in)
        # swin_out = self.swin(swin_in)
        # Encoder
        swin_input_1 = self.E_block1(swin_in)  # 32
        swin_input_1 = self.PPM1(swin_input_1)

        swin_input_2 = self.E_block2(swin_input_1)  # 64
        swin_input_2 = self.PPM2(swin_input_2)

        swin_input_3 = self.E_block3(swin_input_2)  # 128
        swin_input_3 = self.PPM3(swin_input_3)

        swin_input_4 = self.E_block4(swin_input_3)  # 256
        swin_input_4 = self.PPM4(swin_input_4)
        # swin_input_5=self.E_block5(swin_input_4)#512
        # import pdb
        # pdb.set_trace()

        upsample1 = self._block1(swin_input_4)  # 256

        beta_1 = self.conv1_1(swin_out_1[2])
        gamma_1 = self.conv1_2(swin_out_1[2])
        swin_input_3_refine = self.IN_3(swin_input_3) * beta_1 + gamma_1  # 128
        # print('swin_input_3_refine:',swin_input_3_refine.shape) #[4, 256, 128, 128]
        concat3 = torch.cat((swin_input_3, swin_input_3_refine, upsample1), dim=1)  # 256+256+256==768
        # print('concat3:', concat3.shape)  # [4, 768, 128, 128]
        decoder_3 = self.ReLU(self.conv1(concat3))  # 256
        upsample3 = self._block3(decoder_3)  # 128
        upsample3 = self.MSRB2(upsample3)
        # print('upsample3:', upsample3.shape) #[4, 128, 256, 256]

        beta_2 = self.conv2_1(swin_out_1[1])
        gamma_2 = self.conv2_2(swin_out_1[1])
        swin_input_2_refine = self.IN_2(swin_input_2) * beta_2 + gamma_2  # 64
        # print('swin_input_2_refine:', swin_input_2_refine.shape) #[4, 128, 256, 256]
        concat2 = torch.cat((swin_input_2, swin_input_2_refine, upsample3), dim=1)  # 128+128+128=384
        decoder_2 = self.ReLU(self.conv2(concat2))  # 128
        upsample4 = self._block4(decoder_2)  # 64
        upsample4 = self.MSRB3(upsample4)

        beta_3 = self.conv3_1(swin_out_1[0])
        gamma_3 = self.conv3_2(swin_out_1[0])
        swin_input_1_refine = self.IN_1(swin_input_1) * beta_3 + gamma_3  # 32
        # print('swin_input_1_refine:', swin_input_1_refine.shape) #[4, 64, 512, 512]
        concat1 = torch.cat((swin_input_1, swin_input_1_refine, upsample4), dim=1)  # 64+64+64=192
        decoder_1 = self.ReLU(self.conv3(concat1))  # 64
        upsample5 = self._block5(decoder_1)  # 32

        # print('in:', x.shape) #[4,3,1024,1024]

        x6 = self.avg_pool(concat1)
        x5 = self.lf(x6)
        x7 = self.maxpool(concat2)
        x8 = self.maxpool(concat3)
        # print('out:',x.shape) #[4,6,1024,1024]
        return tuple([x5, x6, x7, x8])

class CA(nn.Module):
    def __init__(self, encoder_channels=(96, 192, 384, 768), atrous_rates=(6, 12), num_classes=6, dropout=0.05, bias=False):
        super(CA, self).__init__()
        rate_1, rate_2 = tuple(atrous_rates)
        self.conv_4 = Conv(encoder_channels[3], encoder_channels[3], kernel_size=1)
        self.conv_3 = Conv(encoder_channels[2], encoder_channels[2], kernel_size=1)
        self.conv_2 = Conv(encoder_channels[1], encoder_channels[1], kernel_size=1)
        self.conv_1 = Conv(encoder_channels[0], encoder_channels[0], kernel_size=1)
        self.up42 = nn.Sequential(SeparableConvBNReLU(encoder_channels[-1], encoder_channels[-2], dilation=rate_1),
                                 nn.UpsamplingNearest2d(scale_factor=2),
                                 SeparableConvBNReLU(encoder_channels[-2], encoder_channels[-3], dilation=rate_2),
                                 nn.UpsamplingNearest2d(scale_factor=2))
        self.up31 = nn.Sequential(SeparableConvBNReLU(encoder_channels[-2], encoder_channels[-3], dilation=rate_1),
                                 nn.UpsamplingNearest2d(scale_factor=2),
                                 SeparableConvBNReLU(encoder_channels[-3], encoder_channels[-4], dilation=rate_2),
                                 nn.UpsamplingNearest2d(scale_factor=2))


        self.down24 = DownConnection(encoder_channels[1], encoder_channels[3])
        self.down13 = DownConnection(encoder_channels[0], encoder_channels[2])


    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8):



        # Decoder

        out4 = self.conv_4(x4) + self.down24(x6) # [4, 768, 32, 32]
        # print('out4:', out4.shape)#[4, 768, 32, 32]
        # print('self.conv_3(x3):', self.conv_3(x3).shape)

        out3 = self.conv_3(x3) + self.down13(x5)  #[4, 384, 64, 64]
        # out3 = self.maxpool(self.maxpool(concat2))
        out2 = self.conv_2(x2) + self.up42(x8) #[4, 192, 128, 128]
        # out2 = self.maxpool(self.maxpool(concat1))

        out1 = self.conv_1(x1) + self.up31(x7) #[4, 96, 256, 256]



        return out1, out2, out3, out4



class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(96, 192, 384, 768),
                 dropout=0.05,
                 atrous_rates=(6, 12),
                 num_classes=6):
        super(Decoder, self).__init__()
        self.ca = CA(encoder_channels, atrous_rates)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        self.segmentation_head = nn.Sequential(
            ConvBNReLU(encoder_channels[0], encoder_channels[0]),
            Conv(encoder_channels[0], num_classes, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=4))
        self.up1 = nn.Sequential(
            ConvBNReLU(encoder_channels[1], encoder_channels[0]),
            nn.UpsamplingNearest2d(scale_factor=2)
        )
        self.up2 = nn.Sequential(
            ConvBNReLU(encoder_channels[2], encoder_channels[0]),
            nn.UpsamplingNearest2d(scale_factor=4)
        )
        self.up3 = nn.Sequential(
            ConvBNReLU(encoder_channels[3], encoder_channels[0]),
            nn.UpsamplingNearest2d(scale_factor=8)
        )
        self.init_weight()

    def forward(self,x1, x2, x3, x4, x5, x6, x7, x8):
        out1, out2, out3, out4 = self.ca(x1, x2, x3, x4, x5, x6, x7, x8)
        x = out1 + self.up1(out2) + self.up2(out3) + self.up3(out4)
        x = self.dropout(x)
        # print('1:', x.shape) #[4,96,256,256]
        x = self.segmentation_head(x)
        # print('2:', x.shape) #[4,6,1024,1024]
        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class LMASwin(nn.Module):
    def __init__(self,
                 encoder_channels=(96, 192, 384, 768),
                 dropout=0.05,
                 atrous_rates=(6, 12),
                 num_classes=6,
                 embed_dim=128,
                 depths=(2, 2, 18, 2),
                 num_heads=(4, 8, 16, 32),
                 frozen_stages=2):
        super(LMASwin, self).__init__()
        self.backbone = SwinTransformer(embed_dim=embed_dim, depths=depths, num_heads=num_heads, frozen_stages=frozen_stages)
        self.supencoder = FM(encoder_channels=(96, 192, 384, 768), atrous_rates=(6, 12), num_classes=6, dropout=0.05, bias=False)
        self.decoder = Decoder(encoder_channels, dropout, atrous_rates, num_classes)

    def forward(self, x):
        # print('in:', x.shape) #[4,3,1024,1024]
        x1, x2, x3, x4 = self.backbone(x)
        x5, x6, x7, x8 = self.supencoder(x)
        x = self.decoder(x1, x2, x3, x4, x5, x6, x7, x8)
        # print('out:',x.shape) #[4,6,1024,1024]
        return x

def lmaswin_small(pretrained=True, num_classes=8, weight_path='pretrain_weights/stseg_small.pth'):
    model = LMASwin(encoder_channels=(96, 192, 384, 768),
                   num_classes=num_classes,
                   embed_dim=96,
                   depths=(2, 2, 18, 2),
                   num_heads=(3, 6, 12, 24),
                   frozen_stages=2)

    if pretrained and weight_path is not None:
        old_dict = torch.load(weight_path)['state_dict']
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model