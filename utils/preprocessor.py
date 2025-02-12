import torch
import torch.nn as nn

from network.dataset import RawFrameGroup, GB_Type, FrameGroup
from .tonemap import BaseTonemapper
from utils.masks import Mask
from .image_utils import ImageUtils
from .brdf import BRDFProcessor
from .wavelet import WaveletProcessor

from typing import Dict, Any, Union, Tuple

class Preprocessor:
    def __init__(
        self,
        reconstruction_frame_type: str,
        log_scale_pre_tonemapped: bool,
        pre_tonemapped_normalization_factor: float,
        tonemapper: str,
        log_scale_irridiance: bool,
        irridiance_normalization_factor: float,
        spatial_mask_threasholds: Dict[str, float] = {
            'depth': 0.04,
            'normal': 0.4,
            'albedo': 0.1,
        }
    ):
        super(Preprocessor, self).__init__()
        self.reconstruction_frame_type = reconstruction_frame_type
        self.log_scale_pre_tonemapped = log_scale_pre_tonemapped
        self.pre_tonemapped_normalization_factor = pre_tonemapped_normalization_factor
        self.tonemapper_name = tonemapper
        self.tonemapper = BaseTonemapper.from_name(tonemapper)
        self.log_scale_irridiance = log_scale_irridiance
        self.irridiance_normalization_factor = irridiance_normalization_factor
        self.spatial_mask_threasholds = spatial_mask_threasholds

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
            if self.log_scale_pre_tonemapped:
                hr = BRDFProcessor.exponential_normalize(hr / self.pre_tonemapped_normalization_factor)
                lr = BRDFProcessor.exponential_normalize(lr / self.pre_tonemapped_normalization_factor)
                temporal = BRDFProcessor.exponential_normalize(temporal / self.pre_tonemapped_normalization_factor)
        
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
            if self.log_scale_pre_tonemapped:
                hr = BRDFProcessor.exponential_normalize(hr / self.pre_tonemapped_normalization_factor)
                lr = BRDFProcessor.exponential_normalize(lr / self.pre_tonemapped_normalization_factor)
                temporal = BRDFProcessor.exponential_normalize(temporal / self.pre_tonemapped_normalization_factor)
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
            res['Pred_PreTonemapped'] = reconstructed
            if self.log_scale_pre_tonemapped:
                res['Pred_PreTonemapped'] = BRDFProcessor.exponential_denormalize(res['Pred_PreTonemapped'] * self.pre_tonemapped_normalization_factor)
            res['Pred'] = self.tonemapper(reconstructed)
            res['Pred_PreTonemapped'] = BRDFProcessor.exponential_normalize(res['Pred_PreTonemapped'])
            final = res['Pred']
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
            log_scale_pre_tonemapped=config['log_scale_pre_tonemapped'],
            pre_tonemapped_normalization_factor=config['pre_tonemapped_normalization_factor'],
            tonemapper=config['tonemapper'],
            log_scale_irridiance=config['log_scale_irridiance'],
            irridiance_normalization_factor=config['irridiance_normalization_factor'],
            spatial_mask_threasholds=config['spatial_mask_threasholds']
        )
