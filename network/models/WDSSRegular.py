import torch
import torch.nn as nn

from .ModelBase import ModelBase
from ..modules import *
from utils import *

from datetime import datetime

from typing import Tuple, Dict, Any

class WDSSRegular(ModelBase):
    def __init__(
        self,
    ):
        super(WDSSRegular, self).__init__()

        self.lr_feat_extractor = LRFrameFeatureExtractor(
            12,
            64,
            [32, 48, 48]
        )
        self.temporal_feat_extractor = TemporalFeatExtractor(
            32,
            16,
            [32, 32]
        )
        self.hr_gb_feat_extractor = GBFeatureExtractorDoubleResidual(
            48,
            64,
            3,
            64
        )
        self.feature_fusion = FeatureFusion(
            80,
            12,
            [64, 48]
        )
        self.fminr = FMINRRelu(
            lr_feat_c=32,
            gb_feat_c=32,
            out_c=12,
            mlp_inp_c=64,
            mlp_layer_count=4,
            mlp_layer_size=64
        )
        self.final_wavelet_conv = nn.Conv2d(12, 12, kernel_size=3, padding=1, stride=1)
        self.final_image_conv = nn.Conv2d(3, 3, kernel_size=3, padding=1, stride=1)

    def forward(self, lr_frame, hr_gbuffer, temporal, upscale_factor):
        # Pixel unshuffle
        lr_frame_ps = F.pixel_unshuffle(lr_frame, 2)
        hr_gbuffer_ps = F.pixel_unshuffle(hr_gbuffer, 2)
        temporal_ps = F.pixel_unshuffle(temporal, 2)

        # Extract features
        lr_frame_feat: torch.Tensor = self.lr_feat_extractor(lr_frame_ps)
        temporal_feat: torch.Tensor = self.temporal_feat_extractor(temporal_ps)
        hr_gb_feat: torch.Tensor = self.hr_gb_feat_extractor(hr_gbuffer_ps)

        # Split features for fusion and INR
        lr_ff, lr_inr = torch.split(lr_frame_feat, lr_frame_feat.shape[1]//2, dim=1)
        gb_ff, gb_inr = torch.split(hr_gb_feat, hr_gb_feat.shape[1]//2, dim=1)

        # Feature fusion
        # Upsample lr_ff
        lr_ff = ImageUtils.upsample(lr_ff, upscale_factor)
        fusion_feat = self.feature_fusion.forward(torch.cat([lr_ff, gb_ff, temporal_feat], dim=1))
        # INR
        inr_out = self.fminr.forward(lr_inr, gb_inr, upscale_factor)
        # Combine features
        wavelet_out = fusion_feat + inr_out
        # Final convolution
        wavelet_out = self.final_wavelet_conv(wavelet_out)

        # Inverse wavelet transform
        image = WaveletProcessor.batch_iwt(wavelet_out)
        image = self.final_image_conv(image).clamp(min=0.0, max=None)

        return wavelet_out, image
    
    def timed_forward(self, lr_frame: torch.Tensor, hr_gbuffer: torch.Tensor, temporal: torch.Tensor, upscale_factor: float) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pixel unshuffle
        start = datetime.now()
        lr_frame_ps = F.pixel_unshuffle(lr_frame, 2)
        hr_gbuffer_ps = F.pixel_unshuffle(hr_gbuffer, 2)
        temporal_ps = F.pixel_unshuffle(temporal, 2)
        torch.cuda.synchronize()
        end = datetime.now()
        print(f"Pixel unshuffle time: {(end - start).total_seconds() * 1000:.2f} ms")

        

        # Extract features
        start = datetime.now()
        lr_frame_feat: torch.Tensor = self.lr_feat_extractor(lr_frame_ps)
        torch.cuda.synchronize()
        end = datetime.now()
        print(f"LR Frame feat time: {(end - start).total_seconds() * 1000:.2f} ms")

        start = datetime.now()
        temporal_feat: torch.Tensor = self.temporal_feat_extractor(temporal_ps)
        torch.cuda.synchronize()
        end = datetime.now()
        print(f"Temporal feat time: {(end - start).total_seconds() * 1000:.2f} ms")

        start = datetime.now()
        hr_gb_feat: torch.Tensor = self.hr_gb_feat_extractor(hr_gbuffer_ps)
        torch.cuda.synchronize()
        end = datetime.now()
        print(f"GB Feature: {(end - start).total_seconds() * 1000:.2f} ms")

        # Split features for fusion and INR
        start = datetime.now()
        lr_ff, lr_inr = torch.split(lr_frame_feat, lr_frame_feat.shape[1]//2, dim=1)
        gb_ff, gb_inr = torch.split(hr_gb_feat, hr_gb_feat.shape[1]//2, dim=1)
        torch.cuda.synchronize()
        end = datetime.now()
        print(f"Feature split time: {(end - start).total_seconds() * 1000:.2f} ms")

        # Feature fusion
        # Upsample lr_ff
        start = datetime.now()
        lr_ff = ImageUtils.upsample(lr_ff, upscale_factor)
        fusion_feat = self.feature_fusion.forward(torch.cat([lr_ff, gb_ff, temporal_feat], dim=1))
        torch.cuda.synchronize()
        end = datetime.now()
        print(f"Feature fusion time: {(end - start).total_seconds() * 1000:.2f} ms")
        # INR
        start = datetime.now()
        inr_out = self.fminr.forward(lr_inr, gb_inr, upscale_factor)
        torch.cuda.synchronize()
        end = datetime.now()
        print(f"INR time: {(end - start).total_seconds() * 1000:.2f} ms")

        # Combine features
        wavelet_out = fusion_feat + inr_out
        # Final convolution
        wavelet_out = self.final_wavelet_conv(wavelet_out)

        # Inverse wavelet transform
        torch.cuda.synchronize()
        start = datetime.now()
        image = WaveletProcessor.batch_iwt(wavelet_out)
        torch.cuda.synchronize()
        end = datetime.now()
        print(f"IWT Time: {(end - start).total_seconds() * 1000:.2f} ms")
        image = self.final_image_conv(image).clamp(min=0.0, max=None)

        return wavelet_out, image

