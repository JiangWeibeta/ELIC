import torch
import torch.nn as nn
import torch.nn.functional as F


class EntropyParametersEX(nn.Module):
    def __init__(self, in_dim, out_dim, act=nn.GELU) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 2 // 3 + out_dim * 1 // 3, 1),
            act(),
            nn.Conv2d(in_dim * 2 // 3 + out_dim * 1 // 3, in_dim * 1 // 3 + out_dim * 2 // 3, 1),
            act(),
            nn.Conv2d(in_dim * 1 // 3 + out_dim * 2 // 3, out_dim, 1),
        )

    def forward(self, params):
        """
        Args:
            params(Tensor): [B, C * K, H, W]
        return:
            gaussian_params(Tensor): [B, C * 2, H, W]
        """
        gaussian_params = self.fusion(params)

        return gaussian_params

