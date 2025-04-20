import torch
import torch.nn as nn

from .relu import ReLUINR
from .fourier_mapping import FourierMapping

from typing import List, Dict, Any

class FMINRRelu(nn.Module):
    def __init__(
        self,
        lr_feat_c: int = 32,
        gb_feat_c: int = 32,
        out_c: int = 12,
        mlp_inp_c: int = 64,
        mlp_layer_count: int = 4,
        mlp_layer_size: int = 64
    ):
        super(FMINRRelu, self).__init__()

        self.mapping = FourierMapping(lr_feat_c, gb_feat_c, mlp_inp_c)
        self.mlp = ReLUINR(mlp_inp_c, out_c, mlp_layer_count, mlp_layer_size)
        self.exit_conv = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, stride=1)

    def forward(self, lr_feat: torch.Tensor, gb_feat: torch.Tensor, upscale_factor: float) -> torch.Tensor:
        mapped_feat = self.mapping.forward(lr_feat, gb_feat, upscale_factor)
        out = self.mlp.forward(mapped_feat)
        return self.exit_conv(out)