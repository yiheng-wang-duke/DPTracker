import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class AdaptorBlock(nn.Module):
    def __init__(self, dim, reduced_dim=None, dropout=0.1):
        super(AdaptorBlock, self).__init__()
        if reduced_dim is None:
            reduced_dim = max(dim // 12, 64)
        self.proj_om = nn.Sequential(
            nn.Linear(dim, reduced_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(reduced_dim, dim)
        )
        self.proj_oa = nn.Sequential(
            nn.Linear(dim, reduced_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(reduced_dim, dim)
        )
        
        self.weight_x_om = nn.Parameter(torch.tensor(0.001, dtype=torch.float32))
        self.weight_z_om = nn.Parameter(torch.tensor(0.001, dtype=torch.float32))
        self.weight_x_oa = nn.Parameter(torch.tensor(0.001, dtype=torch.float32))
        self.weight_z_oa = nn.Parameter(torch.tensor(0.001, dtype=torch.float32))
        self.norm_om = nn.LayerNorm(dim)
        self.norm_oa = nn.LayerNorm(dim)

    def forward(self, x, z):
        """
        x: patch embedding, torch.tensor (B,N,C), 
        z: illumination embedding, torch.tensor (B,N,C)
        """
        om = self.norm_om(self.proj_om(x - z))
        oa = self.norm_oa(self.proj_oa(x + z))
        x = x + self.weight_x_oa * oa + self.weight_x_om * om 
        z = z + self.weight_z_oa * oa + self.weight_z_om * om 
        return x, z
        