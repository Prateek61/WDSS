import torch
import torch.nn as nn

from .ModelBase import ModelBase
from ..modules import *
from utils import *

from datetime import datetime

from typing import List, Dict, Any

def get_wdss_model(config: Dict[str, Any]) -> ModelBase:
    if config['name'] == 'WDSS' and config['version'] == 1.0:
        return WDSS.from_config(config)
    else:
        assert False, f"Unknown config: {config}"

class WDSS(ModelBase):
    def __init__(
        self,
        sum_lr_wavelet: bool,
        has_feature_fusion: bool,
        has_fminr: bool,
        lr_feat_extractor_config: Dict[str, Any],
        temporal_feat_extractor_config: Dict[str, Any],
        hr_gb_feat_extractor_config: Dict[str, Any],
        feature_fusion_config: Dict[str, Any],
        fminr_config: Dict[str, Any],
    ):
        super(WDSS, self).__init__()

        self.lr_frame_feat_extractor = BaseLRFeatExtractor.from_config(lr_feat_extractor_config)
        self.temporal_feat_extractor = BaseTemporalFeatExtractor.from_config(temporal_feat_extractor_config)
        self.hr_gb_feat_extractor = BaseGBFeatExtractor.from_config(hr_gb_feat_extractor_config)

        self.has_fminr = has_fminr
        self.has_feature_fusion = has_feature_fusion
        self.sum_lr_wavelet = sum_lr_wavelet

        if self.has_fminr:
            self.fminr = get_fminr(fminr_config)
        if self.has_feature_fusion:
            self.fusion = BaseFeatureFusion.from_config(feature_fusion_config)

        self.final_conv = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=3, padding=1, stride=1)
        )

    def forward(self, lr_frame: torch.Tensor, hr_gbuffer: torch.Tensor, temporal: torch.Tensor, upscale_factor: float) -> torch.Tensor:
        # Pixel unshuffle
        lr_frame_ps = F.pixel_unshuffle(lr_frame, 2)
        hr_gbuffer_ps = F.pixel_unshuffle(hr_gbuffer, 2)
        temporal_ps = F.pixel_unshuffle(temporal, 2)

        # Extract features
        lr_frame_feat = self.lr_frame_feat_extractor(lr_frame_ps)

        temporal_feat = self.temporal_feat_extractor(temporal_ps)

        hr_gb_feat = self.hr_gb_feat_extractor(hr_gbuffer_ps)

        if self.has_fminr and self.has_feature_fusion:
            lr_ff, lr_inr = torch.split(lr_frame_feat, lr_frame_feat.shape[1]//2, dim=1)
            gb_ff, gb_inr = torch.split(hr_gb_feat, hr_gb_feat.shape[1]//2, dim=1)
        elif self.has_feature_fusion:
            lr_ff = lr_frame_feat
            gb_ff = hr_gb_feat
        else:
            lr_inr = lr_frame_feat
            gb_inr = hr_gb_feat

        wavelet_out: torch.Tensor | None = None

        if self.has_fminr:
            wavelet_out = self.fminr.forward(lr_inr, gb_inr, upscale_factor)

        if self.has_feature_fusion:
            lr_ff_upsampled = ImageUtils.upsample(lr_ff, upscale_factor)
            ff_out = self.fusion(torch.cat([lr_ff_upsampled, gb_ff, temporal_feat], dim=1))

            if self.has_fminr:
                wavelet_out = wavelet_out + ff_out
            else:
                wavelet_out = ff_out
        
        if self.sum_lr_wavelet:
            lr_wavelet = WaveletProcessor.batch_wt(lr_frame)
            lr_wavelet_ups = ImageUtils.upsample(lr_wavelet, upscale_factor)
            wavelet_out = wavelet_out + lr_wavelet_ups

        wavelet_out = self.final_conv(wavelet_out)

        image = WaveletProcessor.batch_iwt(wavelet_out).clamp(min=0.0, max=None)

        return wavelet_out, image
    
    @staticmethod
    def from_config(config: Dict[str, Any]) -> 'WDSS':
        return WDSS(
            config['sum_lr_wavelet'],
            config['has_feature_fusion'],
            config['has_fminr'],
            config['lr_feat_extractor'],
            config['temporal_feat_extractor'],
            config['hr_gb_feat_extractor'],
            config['feature_fusion'],
            config['fminr']
        )  
