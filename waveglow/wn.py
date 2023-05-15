import torch
from torch import nn
from torch.nn.utils.weight_norm import remove_weight_norm, weight_norm

from waveglow import fused

TensorTuple = tuple[torch.Tensor, ...]


def Conv1dWN(*args, **kwargs) -> nn.Conv1d:
    return weight_norm(nn.Conv1d(*args, **kwargs))


class GatedResBlock(nn.Module):
    def __init__(self, index: int, n_channels: int, kernel_size=3, last=False):
        super().__init__()

        self.index = index
        dilation = 2**index
        pad = (kernel_size - 1) * dilation // 2
        self.n_channels = n_channels
        self.n_channels_tensor = torch.IntTensor([n_channels])
        self.spec_offset = 2 * n_channels * index
        self.last = last

        self.conv = Conv1dWN(
            n_channels,
            2 * n_channels,
            kernel_size,
            dilation=dilation,
            padding=pad,
        )
        res_channels = n_channels if last else 2 * n_channels
        self.residual = Conv1dWN(n_channels, res_channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, block_input: TensorTuple) -> TensorTuple:
        x, out, spec = block_input

        x = fused.gated_tanh(
            self.conv(x),
            spec[:, self.spec_offset : self.spec_offset + 2 * self.n_channels, :],
            self.n_channels_tensor,
        )

        res_out = self.residual(x)

        if self.last:
            return x, out + res_out, spec
        else:
            return (
                x + res_out[:, : self.n_channels, :],
                out + res_out[:, self.n_channels :, :],
                spec,
            )

    def remove_weight_norm(self):
        self.conv = remove_weight_norm(self.conv)
        self.residual = remove_weight_norm(self.residual)


class WN(nn.Module):
    def __init__(self, in_channels: int, n_layers: int, n_channels: int, n_mels: int):
        super().__init__()

        self.n_layers = n_layers
        self.n_channels = n_channels

        self.initial = Conv1dWN(in_channels, n_channels, 1)

        self.final = nn.Conv1d(n_channels, 2 * in_channels, 1)
        self.final.weight.data.zero_()
        if self.final.bias is not None:
            self.final.bias.data.zero_()

        self.condition = Conv1dWN(n_mels, 2 * n_channels * n_layers, 1)

        self.gated_res_blocks = nn.Sequential(
            *[GatedResBlock(i, n_channels, 3, i == n_layers - 1) for i in range(n_layers)]
        )

    def forward(self, x: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        out = torch.zeros_like(x)
        spec = self.condition(spec)
        x, out, spec = self.gated_res_blocks((x, out, spec))
        return self.final(out)

    def remove_weight_norm(self):
        self.initial = remove_weight_norm(self.initial)
        self.condition = remove_weight_norm(self.condition)
        for block in self.gated_res_blocks:
            if isinstance(block, GatedResBlock):
                block.remove_weight_norm()
