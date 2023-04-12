from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils.weight_norm import weight_norm


def Conv1dWN(*args, **kwargs) -> Conv1d:
    return weight_norm(Conv1d(*args, **kwargs))


def ConvTranspose1dWN(*args, **kwargs) -> ConvTranspose1d:
    return weight_norm(ConvTranspose1d(*args, **kwargs))
