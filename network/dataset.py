import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from torch.utils.data import Dataset
from enum import Enum
from utils.image_utils import ImageUtils
import torch
import zipfile
from random import randint

from typing import Dict, List, Tuple
    
# For comressed Dataset
class GB_Type(Enum):
    """G-Buffer types for the dataset.
    """
    BASE_COLOR = 'BaseColor'
    DIFFUSE_COLOR = 'DiffuseColor'
    MOTION_VECTOR = 'MotionVector'
    NoV = 'NoV' # Dot product of normal and view vector
    DEPTH = 'SceneDepth'
    NORMAL = 'WorldNormal'
    METALLIC = 'Metallic'
    ROUGHNESS = 'Roughness'

GBufferChannels = {
    GB_Type.MOTION_VECTOR: [0, 1],
    GB_Type.DEPTH: [0],
    GB_Type.METALLIC: [0],
    GB_Type.ROUGHNESS: [0],
    GB_Type.NoV: [0]
}

class RawFrameGroup(Enum):
    HR = 'HighRes'
    LR = 'LowRes'
    HR_GB = 'HighResGBuffer'
    LR_GB = 'LowResGBuffer'
    TEMPORAL = 'Temporal'

class FrameGroup(Enum):
    """Pre Processed Frame Groups for the model.
    """
    HR = 'HR'
    LR = 'LR'
    GB = 'GB'
    TEMPORAL = 'TEMPORAL'
    HR_WAVELET = 'HR_WAVELET'
    LR_WAVELET = 'LR_WAVELET'

class WDSSDatasetCompressed(Dataset):
    FRAME_PATHS = {
        'HR_FOLDER': 'HighRes',
        'LR_FOLDER': 'LowRes',
        'HR_GB_FOLDER': 'HighResGBuffer',
        'LR_GB_FOLDER': 'LowResGBuffer'
    }

    def __init__(self, root_dir: str, frames_per_zip: int, patch_size: int = 0, upscale_factor: int = 2):
        self.root_dir = root_dir
        self.frames_per_zip = frames_per_zip
        self.compressed_files = os.listdir(root_dir)
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor

        self.total_frames = len(self.compressed_files) * frames_per_zip

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raw_frames = self._get_raw_frames(idx)

        res: Dict[str, torch.Tensor] = {}

        res[FrameGroup.HR.value] = raw_frames[RawFrameGroup.HR]
        res[FrameGroup.LR.value] = raw_frames[RawFrameGroup.LR]
        res[FrameGroup.TEMPORAL.value] = raw_frames[RawFrameGroup.TEMPORAL]

        gb: torch.Tensor = raw_frames[RawFrameGroup.HR_GB][GB_Type.BASE_COLOR]
        gb = torch.cat([gb, raw_frames[RawFrameGroup.HR_GB][GB_Type.NoV]], dim=0)
        gb = torch.cat([gb, raw_frames[RawFrameGroup.HR_GB][GB_Type.DEPTH]], dim=0)
        gb = torch.cat([gb, raw_frames[RawFrameGroup.HR_GB][GB_Type.NORMAL]], dim=0)
        gb = torch.cat([gb, raw_frames[RawFrameGroup.HR_GB][GB_Type.METALLIC]], dim=0)
        gb = torch.cat([gb, raw_frames[RawFrameGroup.HR_GB][GB_Type.ROUGHNESS]], dim=0)

        res[FrameGroup.GB.value] = gb

        return res


    def _get_raw_frames(self, frame_no: int, no_patch: bool = False) -> Dict[RawFrameGroup, Dict[GB_Type, torch.Tensor] | torch.Tensor]:
        """Returns raw frame patches.
        """
        res = {}

        zip_file_idx = frame_no // self.frames_per_zip
        frame_idx = frame_no % self.frames_per_zip
        frame_idx += 2 # Ignore the first frame, and second frame wont have temporal data as first frame is bad

        # Open the zip file
        with zipfile.ZipFile(os.path.join(self.root_dir, self.compressed_files[zip_file_idx]), 'r') as zip_ref:
            base_folder = self._get_base_folder_name(zip_ref)

            res[RawFrameGroup.HR] = self._get_hr_frame(frame_idx, zip_ref, base_folder)
            res[RawFrameGroup.LR] = self._get_lr_frame(frame_idx, zip_ref, base_folder)
            res[RawFrameGroup.HR_GB] = self._get_hr_g_buffers(frame_idx, zip_ref, base_folder)
            res[RawFrameGroup.LR_GB] = self._get_lr_g_buffers(frame_idx, zip_ref, base_folder)
            res[RawFrameGroup.TEMPORAL] = self._get_hr_frame(frame_idx - 1, zip_ref, base_folder)

        if self.patch_size and not no_patch:
            # Get the patch position
            # Which is a random crop from the frame
            _, lr_h, lr_w = res[RawFrameGroup.LR].shape
            lr_window, hr_window = self._get_random_patch_window((lr_h, lr_w), self.upscale_factor, self.patch_size)
            res[RawFrameGroup.HR] = res[RawFrameGroup.HR][:, hr_window[0][0]:hr_window[1][0], hr_window[0][1]:hr_window[1][1]]
            res[RawFrameGroup.LR] = res[RawFrameGroup.LR][:, lr_window[0][0]:lr_window[1][0], lr_window[0][1]:lr_window[1][1]]
            res[RawFrameGroup.TEMPORAL] = res[RawFrameGroup.TEMPORAL][:, hr_window[0][0]:hr_window[1][0], hr_window[0][1]:hr_window[1][1]]
            for gb_type in GB_Type:
                res[RawFrameGroup.HR_GB][gb_type] = res[RawFrameGroup.HR_GB][gb_type][:, hr_window[0][0]:hr_window[1][0], hr_window[0][1]:hr_window[1][1]]
                res[RawFrameGroup.LR_GB][gb_type] = res[RawFrameGroup.LR_GB][gb_type][:, lr_window[0][0]:lr_window[1][0], lr_window[0][1]:lr_window[1][1]]

        return res


    def _get_hr_g_buffers(self, frame_idx: int, zip_ref: zipfile.ZipFile, base_folder: str) -> Dict[GB_Type, torch.Tensor]:
        res = {}
        for gb_type in GB_Type:
            buffer = zip_ref.read(base_folder + self._get_hr_gb_path(frame_idx, gb_type))
            frame = ImageUtils.decode_exr_image_opencv(buffer)
            if gb_type in GBufferChannels:
                frame = frame[:, :, GBufferChannels[gb_type]]
            res[gb_type] = torch.from_numpy(frame).permute(2, 0, 1)

        return res
    

    def _get_lr_g_buffers(self, frame_idx: int, zip_ref: zipfile.ZipFile, base_folder: str) -> Dict[GB_Type, torch.Tensor]:
        res = {}
        for gb_type in GB_Type:
            buffer = zip_ref.read(base_folder + self._get_lr_gb_path(frame_idx, gb_type))
            frame = ImageUtils.decode_exr_image_opencv(buffer)
            if gb_type in GBufferChannels:
                frame = frame[:, :, GBufferChannels[gb_type]]
            res[gb_type] = torch.from_numpy(frame).permute(2, 0, 1)

        return res
    

    def _get_hr_frame(self, frame_idx: int, zip_ref: zipfile.ZipFile, base_folder: str) -> torch.Tensor:
        buffer = zip_ref.read(base_folder + self._get_hr_frame_path(frame_idx))
        frame = ImageUtils.decode_exr_image_opencv(buffer)
        return torch.from_numpy(frame).permute(2, 0, 1)
    

    def _get_lr_frame(self, frame_idx: int, zip_ref: zipfile.ZipFile, base_folder: str) -> torch.Tensor:
        buffer = zip_ref.read(base_folder + self._get_lr_frame_path(frame_idx))
        frame = ImageUtils.decode_exr_image_opencv(buffer)
        return torch.from_numpy(frame).permute(2, 0, 1)


    def _get_hr_frame_path(self, frame_idx: int) -> str:
        res = self.FRAME_PATHS['HR_FOLDER'] + '/'
        res += str(frame_idx).zfill(4) + '.exr'

        return res
    

    def _get_lr_frame_path(self, frame_idx: int) -> str:
        res = self.FRAME_PATHS['LR_FOLDER'] + '/'
        res += str(frame_idx).zfill(4) + '.exr'

        return res
    

    def _get_hr_gb_path(self, frame_idx: int, gb_type: GB_Type) -> str:
        res = self.FRAME_PATHS['HR_GB_FOLDER'] + '/'
        res += gb_type.value + '.' + str(frame_idx).zfill(4) + '.exr'

        return res
    

    def _get_lr_gb_path(self, frame_idx: int, gb_type: GB_Type) -> str:
        res = self.FRAME_PATHS['LR_GB_FOLDER'] + '/'
        res += gb_type.value + '.' + str(frame_idx).zfill(4) + '.exr'

        return res
    

    def _get_base_folder_name(self, zip_file: zipfile.ZipFile) -> str:
        """There is a single folder in the zip file, return its name.
        """

        return zip_file.namelist()[0].split('/')[0] + '/'

    def _get_random_patch_window(self, LowResolution: Tuple[int, int], UpscaleFactor: float, PatchSize: int) -> Tuple[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Returns a random patch window for the given low resolution and upscale factor.

        Returns:
            Tuple containing the patch window (y, x) for (tl, br) in LR and HR.
        """

        lr_h, lr_w = LowResolution
        patch_h, patch_w = PatchSize, PatchSize

        patch_y = randint(0, lr_h - patch_h)
        patch_x = randint(0, lr_w - patch_w)

        hr_patch_y_tl = patch_y * UpscaleFactor
        hr_patch_x_tl = patch_x * UpscaleFactor
        hr_patch_y_br = hr_patch_y_tl + patch_h * UpscaleFactor
        hr_patch_x_br = hr_patch_x_tl + patch_w * UpscaleFactor

        lr_window = ((patch_y, patch_x), (patch_y + patch_h, patch_x + patch_w))
        hr_window = ((hr_patch_y_tl, hr_patch_x_tl), (hr_patch_y_br, hr_patch_x_br))
        return lr_window, hr_window


    def __len__(self):
        return self.total_frames