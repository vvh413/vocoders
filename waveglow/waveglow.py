import torch
from torch import nn

from waveglow import fused
from waveglow.invertible_1x1_conv import Invertible1x1Conv
from waveglow.wn import WN, TensorTuple


class Flow(nn.Module):
    def __init__(
        self,
        n_remaining: int,
        in_channels: int,
        n_layers: int,
        n_channels: int,
        n_mels: int,
    ):
        super().__init__()
        self.iconv = Invertible1x1Conv(n_remaining)
        self.WN = WN(in_channels, n_layers, n_channels, n_mels)

    def forward(self, flow_input: TensorTuple) -> TensorTuple:
        x, spec = flow_input

        x, log_det = self.iconv(x)

        n_half = x.size(1) // 2
        x_a, x_b = x[:, :n_half, :], x[:, n_half:, :]

        wn_out = self.WN(x_a, spec)
        log_s, b = wn_out[:, n_half:, :], wn_out[:, :n_half, :]
        x_b = torch.exp(log_s) * x_b + b

        x = torch.cat((x_a, x_b), 1)
        return x, log_s, log_det


class WaveGlow(nn.Module):
    def __init__(self, n_mels: int, n_flows: int, n_group: int, wn_layers: int, wn_channels: int):
        super().__init__()

        self.n_flows = n_flows
        self.n_group = n_group
        self.n_early_every = 4
        self.n_early_size = 2
        self.iconv = nn.ModuleList()
        self.WN = nn.ModuleList()
        self.upsample = nn.ConvTranspose1d(n_mels, n_mels, 1024, stride=256)

        n_half = n_group // 2
        n_remaining = n_group
        for i in range(n_flows):
            if i % self.n_early_every == 0 and i > 0:
                n_half -= self.n_early_size // 2
                n_remaining -= self.n_early_size
            self.iconv.append(Invertible1x1Conv(n_remaining))
            self.WN.append(WN(n_half, wn_layers, wn_channels, n_mels * n_group))
        self.n_remaining = n_remaining

    def forward(self, x: torch.Tensor, spec: torch.Tensor) -> TensorTuple:
        spec = self.upsample(spec)
        if spec.size(2) > x.size(1):
            spec = spec[:, :, : x.size(1)]
        spec = spec.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spec = spec.contiguous().view(spec.size(0), spec.size(1), -1).permute(0, 2, 1)
        x = x.unfold(1, self.n_group, self.n_group).permute(0, 2, 1)

        device = self.upsample.weight.device
        out = []
        log_s_out = torch.tensor(0.0, device=device)
        log_det_out = torch.tensor(0.0, device=device)

        for i, (iconv, wn) in enumerate(zip(self.iconv, self.WN)):
            if i % self.n_early_every == 0 and i > 0:
                out.append(x[:, : self.n_early_size, :])
                x = x[:, self.n_early_size :, :]

            x, log_det = iconv(x)
            log_det_out += log_det

            n_half = x.size(1) // 2
            x_a, x_b = x[:, :n_half, :], x[:, n_half:, :]

            wn_out = wn(x_a, spec)

            log_s, b = wn_out[:, n_half:, :], wn_out[:, :n_half, :]
            x_b = torch.exp(log_s) * x_b + b
            x = torch.cat((x_a, x_b), 1)
            # x, log_s = fused.affine(x_a, x_b, wn_out)

            log_s_out += log_s.sum()

        out.append(x)
        return torch.cat(out, 1), log_s_out, log_det_out

    def infer(self, spec, sigma=1.0):
        spec = self.upsample(spec)

        time_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]
        spec = spec[:, :, :-time_cutoff]

        spec = spec.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spec = spec.contiguous().view(spec.size(0), spec.size(1), -1).permute(0, 2, 1)

        device = next(self.parameters()).device
        audio = torch.randn(spec.size(0), self.n_remaining, spec.size(2), device=device)

        audio *= sigma
        for i, (iconv, wn) in reversed(list(enumerate(zip(self.iconv, self.WN)))):
            n_half = audio.size(1) // 2
            audio_a, audio_b = audio[:, :n_half, :], audio[:, n_half:, :]

            wn_out = wn(audio_a, spec)

            log_s, b = wn_out[:, n_half:, :], wn_out[:, :n_half, :]
            audio_b = (audio_b - b) / torch.exp(log_s)

            audio = torch.cat((audio_a, audio_b), 1)
            audio = iconv.inversed(audio)

            if i % self.n_early_every == 0 and i > 0:
                z = torch.randn(spec.size(0), self.n_early_size, spec.size(2), device=device)
                audio = torch.cat((sigma * z, audio), 1)

        audio = audio.permute(0, 2, 1).contiguous().view(audio.size(0), -1).data
        return audio

    def remove_weight_norm(self):
        for wn in self.WN:
            wn.remove_weight_norm()
