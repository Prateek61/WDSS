import torch
import torch.nn as nn

from network.dataset import RawFrameGroup, GB_Type, FrameGroup
from .tonemap import BaseTonemapper
from utils.masks import Mask
from .image_utils import ImageUtils
from .brdf import BRDFProcessor
from .wavelet import WaveletProcessor
from .imge_normalizer import BaseImageNormalizer

from typing import Dict, Any, Union, Tuple, List

class Preprocessor:
    def __init__(
        self,
        reconstruction_frame_type: str,
        pre_tonemapped_normalizers: List[Dict[str, Any]], # List of 
        tonemapper: str,
        irridiance_normalizers: List[Dict[str, Any]],
        spatial_mask_threasholds: Dict[str, float] = {
            'depth': 0.04,
            'normal': 0.4,
            'albedo': 0.1,
        }
    ):
        super(Preprocessor, self).__init__()
        self.reconstruction_frame_type = reconstruction_frame_type
        self.tonemapper_name = tonemapper
        self.tonemapper = BaseTonemapper.from_name(tonemapper)
        self.spatial_mask_threasholds = spatial_mask_threasholds

        self.exponential_normalizer = BaseImageNormalizer.from_config({'type': 'exponential'})
        self.pre_tonemapped_normalizers = [BaseImageNormalizer.from_config(config) for config in pre_tonemapped_normalizers]
        self.irridiance_normalizers = [BaseImageNormalizer.from_config(config) for config in irridiance_normalizers]

        try:
            self._precomp = ImageUtils.opencv_image_to_tensor(ImageUtils.load_exr_image_opencv('res/Precomputed.exr')).squeeze(0)
        except Exception as e:
            if reconstruction_frame_type == 'Irridiance' or reconstruction_frame_type == 'IrridianceAlbedo':
                raise e

    def preprocess(self, raw_frames: Dict[RawFrameGroup, Dict[GB_Type, torch.Tensor] | torch.Tensor], upscale_factor: float) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:

        spatial_mask, temporal_mask = self._spatial_and_temporal_mask(raw_frames, upscale_factor)

        res: Dict[str, torch.Tensor | Dict[str, torch.Tensor]] = {}

        extra: Dict[str, torch.Tensor] = {}
        extra['TemporalMask'] = temporal_mask.squeeze(0)
        # extra['SpatialMask'] = spatial_mask.squeeze(0)

        # Construct gb
        gb: torch.Tensor = self._gb(raw_frames, spatial_mask)

        if self.reconstruction_frame_type == 'Final':
            hr = raw_frames[RawFrameGroup.HR]
            lr = raw_frames[RawFrameGroup.LR]
            temporal = raw_frames[RawFrameGroup.TEMPORAL]
        elif self.reconstruction_frame_type == 'PreTonemapped':
            hr = raw_frames[RawFrameGroup.HR_GB][GB_Type.PRE_TONEMAPPED]
            lr = raw_frames[RawFrameGroup.LR_GB][GB_Type.PRE_TONEMAPPED]
            temporal = raw_frames[RawFrameGroup.TEMPORAL_GB][GB_Type.PRE_TONEMAPPED]
            for normalizer in self.pre_tonemapped_normalizers:
                hr = normalizer.normalize(hr)
                lr = normalizer.normalize(lr)
                temporal = normalizer.normalize(temporal)
        
        warped_temporal = Mask.warp_frame(
            frame=temporal.unsqueeze(0),
            motion_vector=raw_frames[RawFrameGroup.HR_GB][GB_Type.MOTION_VECTOR].unsqueeze(0)
        )
        temporal = torch.cat([warped_temporal.squeeze(0), temporal_mask.squeeze(0)], dim=0)

        res[FrameGroup.TEMPORAL.value] = temporal
        res[FrameGroup.GB.value] = gb
        res[FrameGroup.HR.value] = hr
        res[FrameGroup.LR.value] = lr
        res[FrameGroup.EXTRA.value] = extra

        return res
    
    def preprocess_for_inference(self, raw_frames: Dict[RawFrameGroup, Dict[GB_Type, torch.Tensor] | torch.Tensor], upscale_factor: float) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        res: Dict[str, torch.Tensor | Dict[str, torch.Tensor]] = {}
        extra: Dict[str, torch.Tensor] = {}
        inference: Dict[str, torch.Tensor] = {}

        spatial_mask, temporal_mask = self._spatial_and_temporal_mask(raw_frames, upscale_factor)

        extra['TemporalMask'] = temporal_mask.squeeze(0)

        gb: torch.Tensor = self._gb(raw_frames, spatial_mask)

        if self.reconstruction_frame_type == 'Final':
            hr = raw_frames[RawFrameGroup.HR]
            lr = raw_frames[RawFrameGroup.LR]
            temporal = raw_frames[RawFrameGroup.TEMPORAL]
        elif self.reconstruction_frame_type == 'PreTonemapped':
            hr = raw_frames[RawFrameGroup.HR_GB][GB_Type.PRE_TONEMAPPED]
            lr = raw_frames[RawFrameGroup.LR_GB][GB_Type.PRE_TONEMAPPED]
            temporal = raw_frames[RawFrameGroup.TEMPORAL_GB][GB_Type.PRE_TONEMAPPED]
            for normalizer in self.pre_tonemapped_normalizers:
                hr = normalizer.normalize(hr)
                lr = normalizer.normalize(lr)
                temporal = normalizer.normalize(temporal)
        else:
            raise NotImplementedError(f"Reconstruction frame type {self.reconstruction_frame_type} not supported.")
        
        warped_temporal = Mask.warp_frame(
            frame=temporal.unsqueeze(0),
            motion_vector=raw_frames[RawFrameGroup.HR_GB][GB_Type.MOTION_VECTOR].unsqueeze(0)
        )
        temporal = torch.cat([warped_temporal.squeeze(0), temporal_mask.squeeze(0)], dim=0)

        res[FrameGroup.TEMPORAL.value] = temporal
        res[FrameGroup.GB.value] = gb
        res[FrameGroup.HR.value] = hr
        res[FrameGroup.LR.value] = lr
        res[FrameGroup.EXTRA.value] = extra
        res[FrameGroup.INFERENCE.value] = inference

        return res

    # For logging to tensorboard
    def get_log(self, raw_frames: Dict[RawFrameGroup, Dict[GB_Type, torch.Tensor] | torch.Tensor]) -> Dict[str, torch.Tensor]:
        res: Dict[str, torch.Tensor] = {}

        if self.reconstruction_frame_type == 'Final':
            res['HR'] = raw_frames[RawFrameGroup.HR]
            res['LR'] = raw_frames[RawFrameGroup.LR]
            res['HRWavelet'] = WaveletProcessor.wavelet_transform_image(res['HR'])
            res['LRWavelet'] = WaveletProcessor.wavelet_transform_image(res['LR'])
        elif self.reconstruction_frame_type == 'PreTonemapped':
            res['PreTonemappedHR'] = raw_frames[RawFrameGroup.HR_GB][GB_Type.PRE_TONEMAPPED]
            res['PreTonemappedLR'] = raw_frames[RawFrameGroup.LR_GB][GB_Type.PRE_TONEMAPPED]
            res['HR'] = self.tonemapper(res['PreTonemappedHR'])
            res['LR'] = self.tonemapper(res['PreTonemappedLR'])
            res['PreTonemappedHR'] = BRDFProcessor.exponential_normalize(res['PreTonemappedHR'])
            res['PreTonemappedLR'] = BRDFProcessor.exponential_normalize(res['PreTonemappedLR'])
            res['HRWavelet'] = WaveletProcessor.wavelet_transform_image(res['PreTonemappedHR'])
            res['LRWavelet'] = WaveletProcessor.wavelet_transform_image(res['PreTonemappedLR'])
        elif self.reconstruction_frame_type in ['Irridiance', 'IrridianceAlbedo']:
            raise NotImplementedError(f"Reconstruction frame type {self.reconstruction_frame_type} not supported.")

        return res
    
    def postprocess(self, reconstructed: torch.Tensor, inference_buffers: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        res: Dict[str, torch.Tensor] = {}
        final = reconstructed

        if self.reconstruction_frame_type == 'Final':
            res['Pred'] = reconstructed
        elif self.reconstruction_frame_type == 'PreTonemapped':
            pre_tonemapped = reconstructed
            for normalizer in reversed(self.pre_tonemapped_normalizers):
                pre_tonemapped = normalizer.denormalize(pre_tonemapped)
            res['Pred'] = self.tonemapper(pre_tonemapped)
            res['Pred_PreTonemapped'] = self.exponential_normalizer.normalize(pre_tonemapped)
        else:
            raise NotImplementedError(f"Reconstruction frame type {self.reconstruction_frame_type} not supported.")
        
        return final, res

    def _spatial_and_temporal_mask(self, raw_frames: Dict[RawFrameGroup, Dict[GB_Type, torch.Tensor] | torch.Tensor], upscale_factor: float) -> Tuple[torch.Tensor, torch.Tensor]:
        return Mask.spatial_and_temporal_mask(
            hr_base_color=raw_frames[RawFrameGroup.HR_GB][GB_Type.BASE_COLOR].unsqueeze(0),
            hr_normal=raw_frames[RawFrameGroup.HR_GB][GB_Type.NORMAL].unsqueeze(0),
            hr_depth=raw_frames[RawFrameGroup.HR_GB][GB_Type.NoV_Depth][1:2, :, :].unsqueeze(0),
            lr_base_color=raw_frames[RawFrameGroup.LR_GB][GB_Type.BASE_COLOR].unsqueeze(0),
            lr_normal=raw_frames[RawFrameGroup.LR_GB][GB_Type.NORMAL].unsqueeze(0),
            lr_depth=raw_frames[RawFrameGroup.LR_GB][GB_Type.NoV_Depth][1:2, :, :].unsqueeze(0),
            motion_vector=raw_frames[RawFrameGroup.HR_GB][GB_Type.MOTION_VECTOR].unsqueeze(0),
            temporal_hr_base_color=raw_frames[RawFrameGroup.TEMPORAL_GB][GB_Type.BASE_COLOR].unsqueeze(0),
            temporal_hr_normal=raw_frames[RawFrameGroup.TEMPORAL_GB][GB_Type.NORMAL].unsqueeze(0),
            temporal_hr_depth=raw_frames[RawFrameGroup.TEMPORAL_GB][GB_Type.NoV_Depth][1:2, :, :].unsqueeze(0),
            upscale_factor=upscale_factor,
            spatial_threasholds=self.spatial_mask_threasholds
        )
    
    def _gb(self, raw_frames: Dict[RawFrameGroup, Dict[GB_Type, torch.Tensor] | torch.Tensor], spatial_mask: torch.Tensor) -> torch.Tensor:
        gb: torch.Tensor = raw_frames[RawFrameGroup.HR_GB][GB_Type.BASE_COLOR]
        gb = torch.cat([gb, raw_frames[RawFrameGroup.HR_GB][GB_Type.NoV_Depth]], dim=0)
        gb = torch.cat([gb, raw_frames[RawFrameGroup.HR_GB][GB_Type.NORMAL]], dim=0)
        gb = torch.cat([gb, raw_frames[RawFrameGroup.HR_GB][GB_Type.METALLIC_ROUGHNESS_SPECULAR]], dim=0)
        gb = torch.cat([gb, spatial_mask.squeeze(0)], dim=0)
        return gb

    @staticmethod
    def from_config(config: Dict[str, Any]) -> 'Preprocessor':
        return Preprocessor(
            reconstruction_frame_type=config['reconstruction_frame_type'],
            tonemapper=config['tonemapper'],
            spatial_mask_threasholds=config['spatial_mask_threasholds'],
            pre_tonemapped_normalizers=config['pre_tonemapped_normalizers'],
            irridiance_normalizers=config['irridiance_normalizers']
        )
