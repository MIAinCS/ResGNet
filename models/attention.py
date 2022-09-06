import torch
from torch import nn
from torch.nn import functional as F


class CBAM(nn.Module):
    def __init__(self, channel: int, batch_size: int, division: int = 4):
        super().__init__()

        base = batch_size * channel
        self.shared_mlp = nn.Sequential(
            nn.Linear(base, base // division),
            nn.Linear(base // division, base)
        )

        self.conv1 = nn.Conv3d(2, 1, 3, 1, 1)

    def forward(self, x: torch.Tensor):
        N, C, D, H, W = x.shape

        max_pool = F.adaptive_max_pool3d(x, (1, 1, 1))
        avg_pool = F.adaptive_avg_pool3d(x, (1, 1, 1))

        max_pool = self.shared_mlp(max_pool.flatten())
        avg_pool = self.shared_mlp(avg_pool.flatten())

        channel_attention = F.sigmoid(max_pool + avg_pool)
        channel_attention = channel_attention.reshape((N, C, 1, 1, 1))

        x = channel_attention * x

        # pooling along channel dimenstion
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        f = torch.cat([max_pool, avg_pool], dim=1)
        spatial_attention = F.sigmoid(self.conv1(f))

        x = spatial_attention * x
        return x
