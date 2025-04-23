import torch

from network.commons import GB_TYPE, RawFrameGroup, FrameGroup
from .tonemap import BaseTonemapper
from .masks import Mask
from .brdf import BRDFProcessor
from .wavelet import WaveletProcessor
from .imge_normalizer import BaseImageNormalizer
from config import device

from enum import Enum
from typing import Dict, Any, Tuple, List

class ReconstructionFrameType(Enum):
    PRETONEMAP = 'PreTonemap'
    IRRIDIANCE = 'Irridiance'

class Preprocessor:
    def __init__(
        self,
        reconstruction_frame_type: str,
        tonemapper: str,
        spatial_mask_threasholds: Dict[str, float] = {
            'depth': 0.04,
            'normal': 0.4,
            'albedo': 0.1
        }
    ):
        self.reconstruction_frame_type = ReconstructionFrameType(reconstruction_frame_type)
        self.spatial_mask_threasholds = spatial_mask_threasholds
        self.tonemapper = BaseTonemapper.from_name(tonemapper)
        self.exponential_normalizer = BaseImageNormalizer.from_config({'type': 'exponential'}) # For visualization
        self.reinhard_normalizer = BaseImageNormalizer.from_config({'type': 'reinhard'}) # For visualization

    def preprocess(
        self,
        raw_frames: Dict[RawFrameGroup, Dict[GB_TYPE, torch.Tensor]],
    ) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        res: Dict[str, torch.Tensor | Dict[str, torch.Tensor]] = {}

        upscale_factor = float(raw_frames[RawFrameGroup.HR_GB][GB_TYPE.BASE_COLOR_DEPTH].shape[-2]) / float(raw_frames[RawFrameGroup.LR_GB][GB_TYPE.BASE_COLOR_DEPTH].shape[-2])

        spatial_mask, temporal_mask = self._get_temporal_and_spatial_mask(raw_frames, upscale_factor)
        gb_inp = self._construct_gbuffer_input(raw_frames, spatial_mask)
        
        if self.reconstruction_frame_type == ReconstructionFrameType.PRETONEMAP:
            gt, lr_inp, temporal_inp, extra = self._construct_pretonemap_inputs(raw_frames, temporal_mask, upscale_factor)
        elif self.reconstruction_frame_type == ReconstructionFrameType.IRRIDIANCE:
            gt, lr_inp, temporal_inp, extra = self._construct_irridiance_inputs(raw_frames, temporal_mask, upscale_factor)
        else:
            raise ValueError(f"Unknown reconstruction frame type: {self.reconstruction_frame_type}")
        
        res[FrameGroup.GB_INP.value] = gb_inp
        res[FrameGroup.LR_INP.value] = lr_inp
        res[FrameGroup.GT.value] = gt
        res[FrameGroup.TEMPORAL_INP.value] = temporal_inp
        res[FrameGroup.EXTRA.value] = extra
        return res
    
    def postprocess(
        self,
        frame: torch.Tensor,
        extra: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if self.reconstruction_frame_type == ReconstructionFrameType.IRRIDIANCE:
            frame = BRDFProcessor.brdf_remodulate(frame, extra['BRDF'])

        return frame
    
    # For logging to tensorboard
    def get_log(self,
        raw_frames: Dict[RawFrameGroup, Dict[GB_TYPE, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        res: Dict[str, torch.Tensor] = {}

        preprocessed_frame = self.preprocess(raw_frames)

        res['HR'] = self.exponential_normalizer.normalize(preprocessed_frame[FrameGroup.GT.value])
        res['LR'] = self.exponential_normalizer.normalize(preprocessed_frame[FrameGroup.LR_INP.value])

        res['HRWavelet'] = self.exponential_normalizer.normalize(WaveletProcessor.batch_wt(res['HR'].unsqueeze(0).to(device)).squeeze(0)).to(device='cpu')
        res['LRWavelet'] = self.exponential_normalizer.normalize(WaveletProcessor.batch_wt(res['LR'].unsqueeze(0).to(device)).squeeze(0)).to(device='cpu')

        if self.reconstruction_frame_type == ReconstructionFrameType.IRRIDIANCE:
            res['HRRemodulated'] = self.exponential_normalizer.normalize(BRDFProcessor.brdf_remodulate(preprocessed_frame[FrameGroup.GT.value], preprocessed_frame[FrameGroup.EXTRA.value]['BRDF']))
            res['LRRemodulated'] = self.exponential_normalizer.normalize(BRDFProcessor.brdf_remodulate(preprocessed_frame[FrameGroup.LR_INP.value], preprocessed_frame[FrameGroup.EXTRA.value]['BRDF_LR']))

        if self.reconstruction_frame_type == ReconstructionFrameType.IRRIDIANCE:
            res['HRTonemapped'] = self.tonemap(res['HRRemodulated'])
            res['LRTonemapped'] = self.tonemap(res['LRRemodulated'])
        else:
            res['HRTonemapped'] = self.tonemap(res['HR'])
            res['LRTonemapped'] = self.tonemap(res['LR'])

        return res

    def tonemap(
        self,
        frame: torch.Tensor,
    ) -> torch.Tensor:
        """Tonemap the input frame
        """
        return self.tonemapper.forward(frame)
    
    def _construct_gbuffer_input(
        self,
        raw_frames: Dict[RawFrameGroup, Dict[GB_TYPE, torch.Tensor]],
        spatial_mask: torch.Tensor
    ) -> torch.Tensor:
        """Construct GBuffer input for the model
        """
        gb = torch.cat([
            raw_frames[RawFrameGroup.HR_GB][GB_TYPE.BASE_COLOR_DEPTH],
            raw_frames[RawFrameGroup.HR_GB][GB_TYPE.NORMAL_SPECULAR],
            raw_frames[RawFrameGroup.HR_GB][GB_TYPE.MV_ROUGHNESS_NOV][2:4, :, :],
            raw_frames[RawFrameGroup.HR_GB][GB_TYPE.PRETONEMAP_METALLIC][3:4, :, :],
            spatial_mask.squeeze(0)
        ])
        return gb
    
    def _construct_pretonemap_inputs(
        self,
        raw_frames: Dict[RawFrameGroup, Dict[GB_TYPE, torch.Tensor]],
        temporal_mask: torch.Tensor,
        upscale_factor: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Construct inputs for pre-tonemap reconstruction
        """
        extra: Dict[str, torch.Tensor] = {}

        gt = raw_frames[RawFrameGroup.HR_GB][GB_TYPE.PRETONEMAP_METALLIC][0:3, :, :]
        lr = raw_frames[RawFrameGroup.LR_GB][GB_TYPE.PRETONEMAP_METALLIC][0:3, :, :]
        temporal = raw_frames[RawFrameGroup.TEMPORAL_GB][GB_TYPE.PRETONEMAP_METALLIC][0:3, :, :]

        warped_temporal = Mask.warp_frame(
            frame=temporal.unsqueeze(0),
            motion_vector=raw_frames[RawFrameGroup.HR_GB][GB_TYPE.MV_ROUGHNESS_NOV][0:2, :, :].unsqueeze(0)
        )
        temporal = torch.cat([warped_temporal.squeeze(0), temporal_mask.squeeze(0)], dim=0)

        return gt, lr, temporal, extra
    
    def _construct_irridiance_inputs(
        self,
        raw_frames: Dict[RawFrameGroup, Dict[GB_TYPE, torch.Tensor]],
        temporal_mask: torch.Tensor,
        upscale_factor: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Construct inputs for irradiance reconstruction
        """
        extra: Dict[str, torch.Tensor] = {}

        gt_pretonemap = raw_frames[RawFrameGroup.HR_GB][GB_TYPE.PRETONEMAP_METALLIC][0:3, :, :]
        lr_pretonemap = raw_frames[RawFrameGroup.LR_GB][GB_TYPE.PRETONEMAP_METALLIC][0:3, :, :]
        temporal_pretonemap = raw_frames[RawFrameGroup.TEMPORAL_GB][GB_TYPE.PRETONEMAP_METALLIC][0:3, :, :]

        gt_brdf = self._brdf_extranet(raw_frames[RawFrameGroup.HR_GB])
        lr_brdf = self._brdf_extranet(raw_frames[RawFrameGroup.LR_GB])
        temporal_brdf = self._brdf_extranet(raw_frames[RawFrameGroup.TEMPORAL_GB])

        extra['BRDF'] = gt_brdf
        extra['BRDF_LR'] = lr_brdf

        gt_irridiance = BRDFProcessor.brdf_demodulate(gt_pretonemap, gt_brdf)
        lr_irridiance = BRDFProcessor.brdf_demodulate(lr_pretonemap, lr_brdf)
        temporal_irridiance = BRDFProcessor.brdf_demodulate(temporal_pretonemap, temporal_brdf)

        warped_temporal = Mask.warp_frame(
            frame=temporal_irridiance.unsqueeze(0),
            motion_vector=raw_frames[RawFrameGroup.HR_GB][GB_TYPE.MV_ROUGHNESS_NOV][0:2, :, :].unsqueeze(0)
        )
        temporal_inp = torch.cat([warped_temporal.squeeze(0), temporal_mask.squeeze(0)], dim=0)

        return gt_irridiance, lr_irridiance, temporal_inp, extra

    def _get_temporal_and_spatial_mask(
        self,
        raw_frames: Dict[RawFrameGroup, Dict[GB_TYPE, torch.Tensor]],
        upscale_factor: float  
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get temporal and spatial mask for the model
        """
        
        return Mask.spatial_and_temporal_mask(
            hr_base_color=raw_frames[RawFrameGroup.HR_GB][GB_TYPE.BASE_COLOR_DEPTH][0:3, :, :].unsqueeze(0),
            hr_normal=raw_frames[RawFrameGroup.HR_GB][GB_TYPE.NORMAL_SPECULAR][0:3, :, :].unsqueeze(0),
            hr_depth=raw_frames[RawFrameGroup.HR_GB][GB_TYPE.BASE_COLOR_DEPTH][3:4, :, :].unsqueeze(0),
            lr_base_color=raw_frames[RawFrameGroup.LR_GB][GB_TYPE.BASE_COLOR_DEPTH][0:3, :, :].unsqueeze(0),
            lr_normal=raw_frames[RawFrameGroup.LR_GB][GB_TYPE.NORMAL_SPECULAR][0:3, :, :].unsqueeze(0),
            lr_depth=raw_frames[RawFrameGroup.LR_GB][GB_TYPE.BASE_COLOR_DEPTH][3:4, :, :].unsqueeze(0),
            motion_vector=raw_frames[RawFrameGroup.HR_GB][GB_TYPE.MV_ROUGHNESS_NOV][0:2, :, :].unsqueeze(0),
            temporal_hr_base_color=raw_frames[RawFrameGroup.TEMPORAL_GB][GB_TYPE.BASE_COLOR_DEPTH][0:3, :, :].unsqueeze(0),
            temporal_hr_normal=raw_frames[RawFrameGroup.TEMPORAL_GB][GB_TYPE.NORMAL_SPECULAR][0:3, :, :].unsqueeze(0),
            temporal_hr_depth=raw_frames[RawFrameGroup.TEMPORAL_GB][GB_TYPE.BASE_COLOR_DEPTH][3:4, :, :].unsqueeze(0),
            upscale_factor=upscale_factor,
            spatial_threasholds=self.spatial_mask_threasholds
        )
    
    def _brdf_extranet(
        self,
        gb: Dict[GB_TYPE, torch.Tensor]
    ) -> torch.Tensor:
        return BRDFProcessor.compute_brdf_extranet(
            base_color=gb[GB_TYPE.BASE_COLOR_DEPTH][0:3, :, :],
            specular=gb[GB_TYPE.NORMAL_SPECULAR][3:4, :, :],
            metallic=gb[GB_TYPE.PRETONEMAP_METALLIC][3:4, :, :]
        )
    
    @staticmethod
    def from_config(config: Dict[str, Any]) -> 'Preprocessor':
        """Create a Preprocessor from a config dictionary
        """
        return Preprocessor(
            reconstruction_frame_type=config['reconstruction_frame_type'],
            tonemapper=config['tonemapper'],
            spatial_mask_threasholds=config['spatial_mask_threasholds']
        )
    