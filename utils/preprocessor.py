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
        return self.preprocess_for_inference(raw_frames, upscale_factor)
    
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
        elif self.reconstruction_frame_type in ['Irridiance', 'IrridianceAlbedo', 'IrridianceExtranet']:
            pt_hr = raw_frames[RawFrameGroup.HR_GB][GB_Type.PRE_TONEMAPPED]
            pt_lr = raw_frames[RawFrameGroup.LR_GB][GB_Type.PRE_TONEMAPPED]
            pt_temp = raw_frames[RawFrameGroup.TEMPORAL_GB][GB_Type.PRE_TONEMAPPED]

            for normalizer in self.pre_tonemapped_normalizers:
                pt_hr = normalizer.normalize(pt_hr)
                pt_lr = normalizer.normalize(pt_lr)
                pt_temp = normalizer.normalize(pt_temp)

            if self.reconstruction_frame_type == 'IrridianceAlbedo':
                brdf_map_hr = raw_frames[RawFrameGroup.HR_GB][GB_Type.DIFFUSE_COLOR]
                brdf_map_lr = raw_frames[RawFrameGroup.LR_GB][GB_Type.DIFFUSE_COLOR]
                brdf_map_temporal = raw_frames[RawFrameGroup.TEMPORAL_GB][GB_Type.DIFFUSE_COLOR]
            elif self.reconstruction_frame_type == 'IrridianceExtranet':
                brdf_map_hr = self._brdf_extranet(raw_frames[RawFrameGroup.HR_GB])
                brdf_map_lr = self._brdf_extranet(raw_frames[RawFrameGroup.LR_GB])
                brdf_map_temporal = self._brdf_extranet(raw_frames[RawFrameGroup.TEMPORAL_GB])
            else:
                brdf_map_hr, brdf_map_lr = self._brdf(raw_frames)
                brdf_map_temporal = self._brdf_temporal(raw_frames)

            inference['BRDF_HR'] = brdf_map_hr
            hr = BRDFProcessor.brdf_demodulate(pt_hr, brdf_map_hr)
            lr = BRDFProcessor.brdf_demodulate(pt_lr, brdf_map_lr)
            temporal = BRDFProcessor.brdf_demodulate(pt_temp, brdf_map_temporal)

            for normalizer in self.irridiance_normalizers:
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
        elif self.reconstruction_frame_type in ['PreTonemapped', 'Irridiance', 'IrridianceAlbedo', 'IrridianceExtranet']:
            pt_hr = raw_frames[RawFrameGroup.HR_GB][GB_Type.PRE_TONEMAPPED]
            pt_lr = raw_frames[RawFrameGroup.LR_GB][GB_Type.PRE_TONEMAPPED]

            if self.reconstruction_frame_type in ['Irridiance', 'IrridianceAlbedo', 'IrridianceExtranet']:
                pt_hr_norm = pt_hr
                pt_lr_norm = pt_lr
                for normalizer in self.pre_tonemapped_normalizers:
                    pt_hr_norm = normalizer.normalize(pt_hr_norm)
                    pt_lr_norm = normalizer.normalize(pt_lr_norm)

                if self.reconstruction_frame_type == 'IrridianceAlbedo':
                    brdf_map_hr = raw_frames[RawFrameGroup.HR_GB][GB_Type.DIFFUSE_COLOR]
                    brdf_map_lr = raw_frames[RawFrameGroup.LR_GB][GB_Type.DIFFUSE_COLOR]
                elif self.reconstruction_frame_type == 'IrridianceExtranet':
                    brdf_map_hr = self._brdf_extranet(raw_frames[RawFrameGroup.HR_GB])
                    brdf_map_lr = self._brdf_extranet(raw_frames[RawFrameGroup.LR_GB])
                else:
                    brdf_map_hr, brdf_map_lr = self._brdf(raw_frames)
                
                res['BRDF_HR'] = brdf_map_hr
                res['BRDF_LR'] = brdf_map_lr

                irr_hr = BRDFProcessor.brdf_demodulate(pt_hr_norm, brdf_map_hr)
                irr_lr = BRDFProcessor.brdf_demodulate(pt_lr_norm, brdf_map_lr)

                res['IrridianceHR'] = self.exponential_normalizer.normalize(irr_hr)
                res['IrridianceLR'] = self.exponential_normalizer.normalize(irr_lr)

                pt_hr = BRDFProcessor.brdf_remodulate(irr_hr, brdf_map_hr)
                pt_lr = BRDFProcessor.brdf_remodulate(irr_lr, brdf_map_lr)

                to_wt_hr = irr_hr
                to_wt_lr = irr_lr
                for normalizer in self.irridiance_normalizers:
                    to_wt_hr = normalizer.normalize(to_wt_hr)
                    to_wt_lr = normalizer.normalize(to_wt_lr)
            else:
                to_wt_hr = pt_hr
                to_wt_lr = pt_lr

            res['PreTonemappedHR'] = pt_hr
            res['PreTonemappedLR'] = pt_lr
            res['HR'] = self.tonemapper(pt_hr)
            res['LR'] = self.tonemapper(pt_lr)
            res['HRWavelet'] = WaveletProcessor.wavelet_transform_image(to_wt_hr)
            res['LRWavelet'] = WaveletProcessor.wavelet_transform_image(to_wt_lr)
        else:
            raise NotImplementedError(f"Reconstruction frame type {self.reconstruction_frame_type} not supported.")

        return res
    
    def postprocess(self, reconstructed: torch.Tensor, inference_buffers: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        res: Dict[str, torch.Tensor] = {}
        final = reconstructed
        device = reconstructed.device

        if self.reconstruction_frame_type == 'Final':
            res['Pred'] = reconstructed
        elif self.reconstruction_frame_type == 'PreTonemapped':
            pre_tonemapped = reconstructed
            for normalizer in reversed(self.pre_tonemapped_normalizers):
                pre_tonemapped = normalizer.denormalize(pre_tonemapped)
            res['Pred'] = self.tonemapper(pre_tonemapped)
            res['Pred_PreTonemapped'] = self.exponential_normalizer.normalize(pre_tonemapped)
            final = res['Pred']
        elif self.reconstruction_frame_type in ['Irridiance', 'IrridianceAlbedo', 'IrridianceExtranet']:
            irr = reconstructed
            for normalizer in reversed(self.irridiance_normalizers):
                irr = normalizer.denormalize(irr)
            res['Pred_Irridiance'] = self.exponential_normalizer.normalize(irr)
            pt = BRDFProcessor.brdf_remodulate(irr, inference_buffers['BRDF_HR'].to(device))
            for normalizer in reversed(self.pre_tonemapped_normalizers):
                pt = normalizer.denormalize(pt)
            res['Pred_PreTonemapped'] = self.exponential_normalizer.normalize(pt)
            res['Pred'] = self.tonemapper(pt)
            final = res['Pred']
        else:
            raise NotImplementedError(f"Reconstruction frame type {self.reconstruction_frame_type} not supported.")
        
        return final, res
    
    def postprocess_train(self, frame: torch.Tensor, inference_buffers: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.reconstruction_frame_type == 'Final':
            return frame
        elif self.reconstruction_frame_type == 'PreTonemapped':
            pre_tonemapped = frame
            for normalizer in reversed(self.pre_tonemapped_normalizers):
                pre_tonemapped = normalizer.denormalize(pre_tonemapped)
            return self.tonemapper(pre_tonemapped)
        elif self.reconstruction_frame_type in ['Irridiance', 'IrridianceAlbedo', 'IrridianceExtranet']:
            irr = frame
            for normalizer in reversed(self.irridiance_normalizers):
                irr = normalizer.denormalize(irr)
            pt = BRDFProcessor.brdf_remodulate(irr, inference_buffers['BRDF_HR'])
            for normalizer in reversed(self.pre_tonemapped_normalizers):
                pt = normalizer.denormalize(pt)
            return self.tonemapper(pt)
        else:
            raise NotImplementedError(f"Reconstruction frame type {self.reconstruction_frame_type} not supported.")

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
    
    def _brdf(self, raw_frames: Dict[RawFrameGroup, Dict[GB_Type, torch.Tensor] | torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return BRDFProcessor.compute_brdf(
            diffuse=raw_frames[RawFrameGroup.HR_GB][GB_Type.BASE_COLOR],
            roughness = raw_frames[RawFrameGroup.HR_GB][GB_Type.METALLIC_ROUGHNESS_SPECULAR][1:2, :, :],
            metallic = raw_frames[RawFrameGroup.HR_GB][GB_Type.METALLIC_ROUGHNESS_SPECULAR][0:1, :, :],
            specular = raw_frames[RawFrameGroup.HR_GB][GB_Type.METALLIC_ROUGHNESS_SPECULAR][2:3, :, :],
            NoV = raw_frames[RawFrameGroup.HR_GB][GB_Type.NoV_Depth][0:1, :, :],
            precomp = self._precomp,
            max_idx = 511
        ), BRDFProcessor.compute_brdf(
            diffuse=raw_frames[RawFrameGroup.LR_GB][GB_Type.BASE_COLOR],
            roughness = raw_frames[RawFrameGroup.LR_GB][GB_Type.METALLIC_ROUGHNESS_SPECULAR][1:2, :, :],
            metallic = raw_frames[RawFrameGroup.LR_GB][GB_Type.METALLIC_ROUGHNESS_SPECULAR][0:1, :, :],
            specular = raw_frames[RawFrameGroup.LR_GB][GB_Type.METALLIC_ROUGHNESS_SPECULAR][2:3, :, :],
            NoV = raw_frames[RawFrameGroup.LR_GB][GB_Type.NoV_Depth][0:1, :, :],
            precomp = self._precomp,
            max_idx = 511
        )

    def _brdf_temporal(self, raw_frames: Dict[RawFrameGroup, Dict[GB_Type, torch.Tensor] | torch.Tensor]) -> torch.Tensor:
        return BRDFProcessor.compute_brdf(
            diffuse=raw_frames[RawFrameGroup.TEMPORAL_GB][GB_Type.BASE_COLOR],
            roughness = raw_frames[RawFrameGroup.TEMPORAL_GB][GB_Type.METALLIC_ROUGHNESS_SPECULAR][1:2, :, :],
            metallic = raw_frames[RawFrameGroup.TEMPORAL_GB][GB_Type.METALLIC_ROUGHNESS_SPECULAR][0:1, :, :],
            specular = raw_frames[RawFrameGroup.TEMPORAL_GB][GB_Type.METALLIC_ROUGHNESS_SPECULAR][2:3, :, :],
            NoV = raw_frames[RawFrameGroup.TEMPORAL_GB][GB_Type.NoV_Depth][0:1, :, :],
            precomp = self._precomp,
            max_idx = 511
        )
    
    def _brdf_extranet(self, gb: Dict[GB_Type, torch.Tensor]) -> torch.Tensor:
        return BRDFProcessor.compute_brdf_extranet(
            base_color=gb[GB_Type.BASE_COLOR],
            specular=gb[GB_Type.METALLIC_ROUGHNESS_SPECULAR][2:3, :, :],
            metallic=gb[GB_Type.METALLIC_ROUGHNESS_SPECULAR][0:1, :, :]
        )

    @staticmethod
    def from_config(config: Dict[str, Any]) -> 'Preprocessor':
        return Preprocessor(
            reconstruction_frame_type=config['reconstruction_frame_type'],
            tonemapper=config['tonemapper'],
            spatial_mask_threasholds=config['spatial_mask_threasholds'],
            pre_tonemapped_normalizers=config['pre_tonemapped_normalizers'],
            irridiance_normalizers=config['irridiance_normalizers']
        )
