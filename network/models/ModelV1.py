import torch
import torch.nn as nn
import torch.nn.functional as F

from network.modules.feature_extractors import LRFrameFeatureExtractor
from network.modules.reconstruction_modules import FeatureFusion

class WDSSV1(nn.Module):
    def __init__(self):
        super(WDSSV1, self).__init__()

        self.upscale_factor = 2
        self.lr_frame_feature_extractor = LRFrameFeatureExtractor(12, 32, [32, 32])
        self.feature_fusion = FeatureFusion(32, 12, [64, 48])

    def forward(self, lr_frame: torch.Tensor) -> torch.Tensor:
        # Perform space to depth operation with scale factor 2
        lr_frame_unsuffled = F.pixel_unshuffle(lr_frame, 2)

        lr_frame_features = self.lr_frame_feature_extractor(lr_frame_unsuffled)
        # # Bilinear upsample the features
        lr_frame_features_upsampled = F.interpolate(lr_frame_features, scale_factor=self.upscale_factor, mode='bilinear')
        # # Feature fusion
        hr_frame = self.feature_fusion(lr_frame_features_upsampled)

        # # Perform depth to space operation with scale factor 2
        hr_frame = F.pixel_shuffle(hr_frame, 2)

        return hr_frame 