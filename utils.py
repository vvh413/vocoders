from typing import Any

import torch
import inspect
from torch import nn
from torchaudio.transforms import MelSpectrogram


class MelSpec(nn.Module):
    def __init__(self, conf: dict[str, Any]):
        super().__init__()
        conf = conf.copy()
        self.conf = conf.copy()
        self.log_fn = conf.pop("log_fn", torch.log)
        self.norm_fn = conf.pop("norm_fn", None)
        self.featurizer = MelSpectrogram(**conf)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        melspec = self.featurizer(audio).squeeze()
        melspec = self.log_fn(melspec.clamp(1e-5))
        if self.norm_fn:
            melspec = self.norm_fn(melspec)
        return melspec


def lambda2str(fn) -> str:
    if fn is None:
        return "None"
    return inspect.getsourcelines(fn)[0][0].strip()
