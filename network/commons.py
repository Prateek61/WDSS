import torch
import torch.nn.functional as F
from enum import Enum

# Enum to represent activation functions
class ActivationFn(Enum):
    GELU = torch.nn.GELU()
    RELU = torch.nn.ReLU(inplace=True)
    SOFTMAX = torch.nn.Softmax(dim=1)
    SOFTMAX2D = torch.nn.Softmax2d()
    SIGMOID = torch.nn.Sigmoid()
    TANH = torch.nn.Tanh()
    LRELU = torch.nn.LeakyReLU(0.2)

def pad_tensor(t: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
    """
    Pad a tensor with zeros according to a pattern.
    :param t: tensor to pad
    :param pattern: pattern to follow
    :return: padded tensor
    """
    
    pattern = pattern.view(1, -1, 1, 1)
    t = F.pad(t, (1, 1, 1, 1), 'constant', 0)
    t[:, :, 0:1, :] = pattern
    t[:, :, -1:, :] = pattern
    t[:, :, :, 0:1] = pattern
    t[:, :, :, -1:] = pattern

    return t

def get_activation_fn(activation_fn: ActivationFn) -> torch.nn.Module:
    """
    Get the activation function module.
    :param activation_fn: activation function to get
    :return: activation function module
    """
    
    return activation_fn.value