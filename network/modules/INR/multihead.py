import torch 
import torch.nn as nn

from .relu import ReLUINR
from .fourier_mapping import FourierMapping

class MultiHeadINR(nn.Module):
    def __init__(
        self,
        lr_feat_c: int = 32,
        gb_feat_c: int = 32,
        mlp_inp_c: int = 64,
        mlp_layer_count: int = 4,
        mlp_layer_size: int = 42
    ):
        super(MultiHeadINR, self).__init__()

        self.mapping = FourierMapping(lr_feat_c, gb_feat_c, mlp_inp_c)
        
        self.mlp_ll = ReLUINR(mlp_inp_c, 3, mlp_layer_count, mlp_layer_size)
        self.mlp_hl = ReLUINR(mlp_inp_c, 3, mlp_layer_count, mlp_layer_size)
        self.mlp_lh = ReLUINR(mlp_inp_c, 3, mlp_layer_count, mlp_layer_size)
        self.mlp_hh = ReLUINR(mlp_inp_c, 3, mlp_layer_count, mlp_layer_size)
        self.exit_conv = nn.Conv2d(12, 12, kernel_size=3, padding=1, stride=1) 


    def forward(self, lr_feat: torch.Tensor, gb_feat: torch.Tensor, upscale_factor: float) -> torch.Tensor:
        mapped_feat = self.mapping.forward(lr_feat, gb_feat, upscale_factor)
        out_ll = self.mlp_ll.forward(mapped_feat)
        out_hl = self.mlp_hl.forward(mapped_feat)
        out_lh = self.mlp_lh.forward(mapped_feat)
        out_hh = self.mlp_hh.forward(mapped_feat)
        out = torch.cat((out_ll, out_hl, out_lh, out_hh), dim=1)
        return self.exit_conv(out)