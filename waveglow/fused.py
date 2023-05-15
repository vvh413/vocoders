import torch


@torch.jit.script
def gated_tanh(a: torch.Tensor, b: torch.Tensor, n_channels: torch.Tensor) -> torch.Tensor:
    n_channels_int = n_channels[0]
    x = a + b
    return torch.tanh(x[:, :n_channels_int, :]) * torch.sigmoid(x[:, n_channels_int:, :])


@torch.jit.script
def affine(x_a: torch.Tensor, x_b: torch.Tensor, wn_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n_half = x_a.size(1)
    log_s, b = wn_out[:, n_half:, :], wn_out[:, :n_half, :]
    x_b = torch.exp(log_s) * x_b + b
    return torch.cat((x_a, x_b), 1), log_s


@torch.jit.script
def log_det(z: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    batch_size, group_size, n_groups = z.size()
    return batch_size * n_groups * torch.logdet(W.squeeze())
