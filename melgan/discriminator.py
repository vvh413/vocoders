import torch
from torch import nn
from melgan.weight_norm_conv import Conv1dWN


class DiscriminatorBlock(nn.Module):
    def __init__(self, ndf: int, n_layers: int, ratio: int):
        super().__init__()
        n_channels = ndf
        model: list[nn.Module] = [
            nn.Sequential(
                Conv1dWN(
                    1,
                    n_channels,
                    kernel_size=15,
                    padding=7,
                    padding_mode="reflect"
                ),
                nn.LeakyReLU(0.2),
            )
        ]

        for _ in range(n_layers):
            n_channels_next = min(n_channels * ratio, 1024)
            model.append(nn.Sequential(
                Conv1dWN(
                    n_channels,
                    n_channels_next,
                    kernel_size=ratio * 10 + 1,
                    stride=ratio,
                    padding=ratio * 5,
                    groups=n_channels // 4,
                ),
                nn.LeakyReLU(0.2),
            ))
            n_channels = n_channels_next

        model.append(nn.Sequential(
            Conv1dWN(
                n_channels,
                min(n_channels * 2, 1024),
                kernel_size=5,
                padding=2,
            ),
            nn.LeakyReLU(0.2),
        ))
        model.append(Conv1dWN(
            min(n_channels * 2, 1024),
            1,
            kernel_size=3,
            padding=1,
        ))

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        results = []
        for layer in self.model:
            x = layer(x)
            results.append(x)
        return results


class Discriminator(nn.Module):
    def __init__(self, n_disc: int, ndf: int, n_layers: int, ratio: int):
        super().__init__()
        self.model = nn.Sequential(*[
            DiscriminatorBlock(ndf, n_layers, ratio)
            for _ in range(n_disc)
        ])
        self.avg = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> list[list[torch.Tensor]]:
        results = []
        for disc in self.model:
            results.append(disc(x))
            x = self.avg(x)
        return results
