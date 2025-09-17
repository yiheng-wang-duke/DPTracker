import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple
from torchvision.ops import DeformConv2d, deform_conv2d

class DeformConv2dPack(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.offset_conv = nn.Conv2d(
            in_channels, 
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )
        self.deform_conv = DeformConv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        
    def forward(self, x):
        offset = self.offset_conv(x)
        return self.deform_conv(x, offset)


class ViewPrompter(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, reduced_dim=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        assert patch_size == (16, 16), f'received patchsize {patch_size}'
        if reduced_dim is None:
            reduced_dim = max(embed_dim // 8, 64)
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, reduced_dim, kernel_size=4, stride=4),
            nn.BatchNorm2d(reduced_dim),
            nn.LeakyReLU(),
            DeformConv2dPack(reduced_dim, embed_dim, kernel_size=4, stride=4),
            nn.BatchNorm2d(embed_dim),
            nn.LeakyReLU()
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
