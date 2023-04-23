import os
import random

import pandas as pd
import torchaudio
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchaudio.transforms import Resample


class AudioDataset(Dataset):
    EFFECTS = [
        ["gain", "-n"],  # normalises to 0dB
        ["pitch", "5"],  # 5 cent pitch shift
    ]

    def __init__(self, root_dir, sample_rate, augment=True, segment_length=0):
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.root_dir = root_dir
        self.files = os.listdir(root_dir)
        self.augment = augment

    def __getitem__(self, index):
        data, sr = torchaudio.load(os.path.join(self.root_dir, self.files[index]))
        if sr != self.sample_rate:
            resampler = Resample(sr, self.sample_rate)
            data = resampler(data)

        if self.augment:
            data, _ = torchaudio.sox_effects.apply_effects_tensor(
                data, self.sample_rate, self.EFFECTS
            )

        # data = data.squeeze()
        if self.segment_length:
            if data.size(1) >= self.segment_length:
                max_audio_start = data.size(1) - self.segment_length
                audio_start = random.randint(0, max_audio_start)
                data = data[:, audio_start : audio_start + self.segment_length]
            else:
                data = F.pad(
                    data, (0, self.segment_length - data.size(1)), "constant"
                ).data

        return data

    def __len__(self):
        return len(self.files)


class TTSDataset(Dataset):
    EFFECTS = [
        ["gain", "-n"],  # normalises to 0dB
        ["pitch", "5"],  # 5 cent pitch shift
    ]

    def __init__(self, root_dir, df, sample_rate, augment=True, segment_length=0):
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.root_dir = root_dir
        self.df = df
        self.augment = augment

    def __getitem__(self, index):
        row = self.df.iloc[index]
        data, sr = torchaudio.load(os.path.join(self.root_dir, f"{row.audio_id}.wav"))
        if sr != self.sample_rate:
            resampler = Resample(sr, self.sample_rate)
            data = resampler(data)

        if self.augment:
            data, _ = torchaudio.sox_effects.apply_effects_tensor(
                data, self.sample_rate, self.EFFECTS
            )

        # data = data.squeeze()
        if self.segment_length:
            if data.size(1) >= self.segment_length:
                max_audio_start = data.size(1) - self.segment_length
                audio_start = random.randint(0, max_audio_start)
                data = data[:, audio_start : audio_start + self.segment_length]
            else:
                data = F.pad(
                    data, (0, self.segment_length - data.size(1)), "constant"
                ).data

        return data, row.text

    def __len__(self):
        return len(self.df)
