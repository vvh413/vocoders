import torch
from torch import nn

from melgan.weight_norm_conv import Conv1dWN, ConvTranspose1dWN


class ResBlock(nn.Module):
    def __init__(self, n_channels: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            Conv1dWN(
                n_channels,
                n_channels,
                kernel_size=3,
                dilation=dilation,
                padding=dilation,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
            Conv1dWN(n_channels, n_channels, kernel_size=1),
        )
        self.shortcut = Conv1dWN(n_channels, n_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.block(x)


class Generator(nn.Module):
    def __init__(self, n_mels: int, ngf: int, n_res_blocks: int):
        super().__init__()
        ratios = [8, 8, 2, 2]
        n_channels = ngf * int(2 ** len(ratios))
        model = [
            Conv1dWN(
                n_mels,
                n_channels,
                kernel_size=7,
                padding=3,
                padding_mode="reflect",
            ),
        ]

        for ratio in ratios:
            model += [
                nn.LeakyReLU(0.2),
                ConvTranspose1dWN(
                    n_channels,
                    n_channels // 2,
                    kernel_size=2 * ratio,
                    stride=ratio,
                    padding=ratio // 2 + ratio % 2,
                    output_padding=ratio % 2,
                ),
            ]

            model += [
                ResBlock(n_channels // 2, dilation=3**i) for i in range(n_res_blocks)
            ]

            n_channels //= 2

        model += [
            nn.LeakyReLU(0.2),
            Conv1dWN(
                n_channels,
                1,
                kernel_size=7,
                padding=3,
                padding_mode="reflect",
            ),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, melspec: torch.Tensor) -> torch.Tensor:
        return self.model(melspec)
