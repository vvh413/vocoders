import torch
from torch import nn
from torch.nn import functional as F

from waveglow import fused


class Invertible1x1Conv(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv1d(n_channels, n_channels, 1, bias=False)
        W, _ = torch.linalg.qr(torch.Tensor(n_channels, n_channels).normal_())
        if torch.det(W) < 0:
            W[:, 0] *= -1
        self.conv.weight.data = W[..., None]
        self.iW = None

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, group_size, n_groups = z.size()
        W = self.conv.weight.squeeze()
        log_det = batch_size * n_groups * torch.logdet(W)
        # log_det = fused.log_det(z, self.conv.weight)
        return self.conv(z), log_det

    def inversed(self, z: torch.Tensor) -> torch.Tensor:
        if self.iW is None:
            W = self.conv.weight.squeeze()
            self.iW = W.inverse()[..., None]
        return F.conv1d(z, self.iW)
