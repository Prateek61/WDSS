import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from torch.utils.data import Dataset
from enum import Enum
from utils.image_utils import ImageUtils
import torch
import zipfile
from random import randint
import cv2
import numpy as np
from utils.wavelet import WaveletProcessor
from config import Settings
from multiprocessing.pool import ThreadPool, AsyncResult
from utils.masks import Mask
from utils.patch import Patch

import threading

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
    SPECULAR = 'Specular'

GBufferChannels = {
    GB_Type.MOTION_VECTOR: [0, 1],
    GB_Type.DEPTH: [0],
    GB_Type.METALLIC: [0],
    GB_Type.ROUGHNESS: [0],
    GB_Type.NoV: [0],
    GB_Type.SPECULAR: [0]
}

class RawFrameGroup(Enum):
    HR = 'HighRes'
    LR = 'LowRes'
    HR_GB = 'HighResGBuffer'
    LR_GB = 'LowResGBuffer'
    TEMPORAL = 'Temporal'
    TEMPORAL_GB = 'TemporalGBuffer'

class FrameGroup(Enum):
    """Pre Processed Frame Groups for the model.
    """
    HR = 'HR'
    LR = 'LR'
    GB = 'GB'
    TEMPORAL = 'TEMPORAL'

class DatasetUtils:
    @staticmethod
    def get_buffer(zip_ref: zipfile.ZipFile, file_path: str) -> bytes:
        return zip_ref.read(file_path)
    
    @staticmethod
    def get_frame_from_buffer(buffer: bytes) -> torch.Tensor:
        frame = ImageUtils.decode_exr_image_opencv(buffer)
        return torch.from_numpy(frame).permute(2, 0, 1)
    
    @staticmethod
    def get_frame(zip_ref: zipfile.ZipFile, file_path: str) -> torch.Tensor:
        buffer = zip_ref.read(file_path)
        return DatasetUtils.get_frame_from_buffer(buffer)
    
    @staticmethod
    def wrap_try(func):
        def wrapper(*args, **kwargs):
            recursion_depth = kwargs.pop('recursion_depth', 0)

            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f'Error in {func.__name__}: {e}, depth: {recursion_depth}')

                if recursion_depth > 5:
                    return None

                recursion_depth += 1
                kwargs['recursion_depth'] = recursion_depth

                return wrapper(*args, **kwargs)
        return wrapper

class WDSSDatasetBase(Dataset):
    def __init__(self):
        super(WDSSDatasetBase, self).__init__()

    def __getitem__(self, index):
        raise NotImplementedError

class WDSSDatasetCompressed(Dataset):
    FRAME_PATHS = {
        'HR_FOLDER': 'HighRes',
        'LR_FOLDER': 'LowRes',
        'HR_GB_FOLDER': 'HighResGbuffer',
        'LR_GB_FOLDER': 'LowResGbuffer'
    }

    def __init__(self, root_dir: str, frames_per_zip: int, patch_size: int = 0, upscale_factor: int = 2, multi_patches_per_frame: bool = False, num_threads: int = 8):
        self.root_dir = root_dir
        self.frames_per_zip = frames_per_zip
        self.compressed_files = os.listdir(root_dir)
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.multi_patches_per_frame = multi_patches_per_frame
        self.thread_pool = ThreadPool(max(num_threads, 4))

        self.patch = Patch((360, 640), self.upscale_factor, self.patch_size, self.multi_patches_per_frame)

        self.patches_per_frame = self.patch.patches_per_frame
        self.total_frames = len(self.compressed_files) * self.patches_per_frame * frames_per_zip

    @DatasetUtils.wrap_try
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raw_frames = self._get_raw_frames(idx)

        res: Dict[str, torch.Tensor] = {}

        res[FrameGroup.HR.value] = raw_frames[RawFrameGroup.HR]
        res[FrameGroup.LR.value] = raw_frames[RawFrameGroup.LR]
        
        spatial_mask, temporal_mask = Mask.spatial_and_temporal_mask(
            hr_base_color=raw_frames[RawFrameGroup.HR_GB][GB_Type.BASE_COLOR].unsqueeze(0),
            hr_normal=raw_frames[RawFrameGroup.HR_GB][GB_Type.NORMAL].unsqueeze(0),
            hr_depth=raw_frames[RawFrameGroup.HR_GB][GB_Type.DEPTH].unsqueeze(0),
            lr_base_color=raw_frames[RawFrameGroup.LR_GB][GB_Type.BASE_COLOR].unsqueeze(0),
            lr_normal=raw_frames[RawFrameGroup.LR_GB][GB_Type.NORMAL].unsqueeze(0),
            lr_depth=raw_frames[RawFrameGroup.LR_GB][GB_Type.DEPTH].unsqueeze(0),
            motion_vector=raw_frames[RawFrameGroup.HR_GB][GB_Type.MOTION_VECTOR].unsqueeze(0),
            temporal_hr_base_color=raw_frames[RawFrameGroup.TEMPORAL_GB][GB_Type.BASE_COLOR].unsqueeze(0),
            temporal_hr_normal=raw_frames[RawFrameGroup.TEMPORAL_GB][GB_Type.NORMAL].unsqueeze(0),
            temporal_hr_depth=raw_frames[RawFrameGroup.TEMPORAL_GB][GB_Type.DEPTH].unsqueeze(0),
            upscale_factor=2, 
            spatial_threasholds={
                'depth': 0.04,
                'normal': 0.4,
                'albedo': 0.1
            }
        )

        gb: torch.Tensor = raw_frames[RawFrameGroup.HR_GB][GB_Type.BASE_COLOR]
        gb = torch.cat([gb, raw_frames[RawFrameGroup.HR_GB][GB_Type.NoV]], dim=0)
        gb = torch.cat([gb, raw_frames[RawFrameGroup.HR_GB][GB_Type.DEPTH]], dim=0)
        gb = torch.cat([gb, raw_frames[RawFrameGroup.HR_GB][GB_Type.NORMAL]], dim=0)
        gb = torch.cat([gb, raw_frames[RawFrameGroup.HR_GB][GB_Type.METALLIC]], dim=0)
        gb = torch.cat([gb, raw_frames[RawFrameGroup.HR_GB][GB_Type.ROUGHNESS]], dim=0)
        gb = torch.cat([gb, raw_frames[RawFrameGroup.HR_GB][GB_Type.SPECULAR]], dim=0)
        gb = torch.cat([gb, spatial_mask.squeeze(0)], dim=0)
        res[FrameGroup.GB.value] = gb

        warped_temporal = Mask.warp_frame(frame=raw_frames[RawFrameGroup.TEMPORAL].unsqueeze(0), motion_vector=raw_frames[RawFrameGroup.HR_GB][GB_Type.MOTION_VECTOR].unsqueeze(0))
        res[FrameGroup.TEMPORAL.value] = torch.cat([warped_temporal.squeeze(0), temporal_mask.squeeze(0)], dim=0)

        return res

    def _get_raw_frames(self, frame_no: int, no_patch: bool = False) -> Dict[RawFrameGroup, Dict[GB_Type, torch.Tensor] | torch.Tensor]:
        """Returns raw frame patches.
        """
        res: Dict[RawFrameGroup, Dict[GB_Type, torch.Tensor] | torch.Tensor] = {}

        # This is not working inside the with block
        thread_hr_gb: Dict[GB_Type, torch.Tensor] = {}
        thread_lr_gb: Dict[GB_Type, torch.Tensor] = {}
        thread_temporal_gb: Dict[GB_Type, torch.Tensor] = {}

        if not no_patch:
            patch_idx = frame_no % self.patches_per_frame
            frame_no = frame_no // self.patches_per_frame
        zip_file_idx = frame_no // self.frames_per_zip
        frame_idx = frame_no % self.frames_per_zip
        frame_idx += 2 # Ignore the first frame, and second frame wont have temporal data as first frame is bad

        # Open the zip file
        with zipfile.ZipFile(os.path.join(self.root_dir, self.compressed_files[zip_file_idx]), 'r') as zip_ref:
            base_folder = self._get_base_folder_name(zip_ref)

            hr_frame_res = self.thread_pool.apply_async(DatasetUtils.wrap_try(self._get_hr_frame), (frame_idx, zip_ref, base_folder))
            lr_frame_res = self.thread_pool.apply_async(DatasetUtils.wrap_try(self._get_lr_frame), (frame_idx, zip_ref, base_folder))
            temporal_frame_res = self.thread_pool.apply_async(DatasetUtils.wrap_try(self._get_hr_frame), (frame_idx - 1, zip_ref, base_folder))
            hr_gbuffer_res = self.thread_pool.apply_async(DatasetUtils.wrap_try(self._get_hr_g_buffers), (frame_idx, zip_ref, base_folder))
            lr_gbuffer_res = self.thread_pool.apply_async(DatasetUtils.wrap_try(self._get_lr_g_buffers), (frame_idx, zip_ref, base_folder))
            temporal_gbuffer_res = self.thread_pool.apply_async(DatasetUtils.wrap_try(self._get_temporal_g_buffers), (frame_idx, zip_ref, base_folder))

            res[RawFrameGroup.HR_GB] = hr_gbuffer_res.get()
            res[RawFrameGroup.LR_GB] = lr_gbuffer_res.get()
            res[RawFrameGroup.TEMPORAL_GB] = temporal_gbuffer_res.get()
            res[RawFrameGroup.HR] = hr_frame_res.get()
            res[RawFrameGroup.LR] = lr_frame_res.get()
            res[RawFrameGroup.TEMPORAL] = temporal_frame_res.get()

        if self.patch_size and not no_patch:
            # Get the patch position
            # Which is a random crop from the frame
            _, lr_h, lr_w = res[RawFrameGroup.LR].shape
            lr_window, hr_window = self.patch.get_patch_window(patch_idx)
            res[RawFrameGroup.HR] = res[RawFrameGroup.HR][:, hr_window[0][0]:hr_window[1][0], hr_window[0][1]:hr_window[1][1]]
            res[RawFrameGroup.LR] = res[RawFrameGroup.LR][:, lr_window[0][0]:lr_window[1][0], lr_window[0][1]:lr_window[1][1]]
            res[RawFrameGroup.TEMPORAL] = res[RawFrameGroup.TEMPORAL][:, hr_window[0][0]:hr_window[1][0], hr_window[0][1]:hr_window[1][1]]
            for gb_type in GB_Type:
                res[RawFrameGroup.HR_GB][gb_type] = res[RawFrameGroup.HR_GB][gb_type][:, hr_window[0][0]:hr_window[1][0], hr_window[0][1]:hr_window[1][1]]
                res[RawFrameGroup.LR_GB][gb_type] = res[RawFrameGroup.LR_GB][gb_type][:, lr_window[0][0]:lr_window[1][0], lr_window[0][1]:lr_window[1][1]]
            for gb_type in [GB_Type.BASE_COLOR, GB_Type.DEPTH, GB_Type.NORMAL]:
                res[RawFrameGroup.TEMPORAL_GB][gb_type] = res[RawFrameGroup.TEMPORAL_GB][gb_type][:, hr_window[0][0]:hr_window[1][0], hr_window[0][1]:hr_window[1][1]]

        return res

    def _get_gbuffer(self, frame_idx: int, zip_ref: zipfile.ZipFile, base_folder: str, gb_type: GB_Type, path: str) -> torch.Tensor:
        frame: torch.Tensor = DatasetUtils.get_frame(zip_ref, base_folder + path)
        if gb_type in GBufferChannels:
            # frame = frame[:, :, GBufferChannels[gb_type]]
            frame = frame[GBufferChannels[gb_type], :, :]
        return frame

    def _get_hr_g_buffers(self, frame_idx: int, zip_ref: zipfile.ZipFile, base_folder: str) -> Dict[GB_Type, torch.Tensor]:
        res = {}

        results: List[AsyncResult] = []
        for gb_type in GB_Type:
            results.append(self.thread_pool.apply_async(DatasetUtils.wrap_try(self._get_gbuffer), (frame_idx, zip_ref, base_folder, gb_type, self._get_hr_gb_path(frame_idx, gb_type))))

        for gb_type, result in zip(GB_Type, results):
            res[gb_type] = result.get()

        return res
    

    def _get_lr_g_buffers(self, frame_idx: int, zip_ref: zipfile.ZipFile, base_folder: str) -> Dict[GB_Type, torch.Tensor]:
        res = {}

        results: List[AsyncResult] = []
        for gb_type in GB_Type:
            results.append(self.thread_pool.apply_async(DatasetUtils.wrap_try(self._get_gbuffer), (frame_idx, zip_ref, base_folder, gb_type, self._get_lr_gb_path(frame_idx, gb_type))))

        for gb_type, result in zip(GB_Type, results):
            res[gb_type] = result.get()

        return res
    
    def _get_temporal_g_buffers(self, frame_idx: int, zip_ref: zipfile.ZipFile, base_folder: str) -> Dict[GB_Type, torch.Tensor]:
        # Just need the BaseColor, Depth, and Normal
        res = {}

        results: List[AsyncResult] = []
        for gb_type in [GB_Type.BASE_COLOR, GB_Type.DEPTH, GB_Type.NORMAL]:
            results.append(self.thread_pool.apply_async(DatasetUtils.wrap_try(self._get_gbuffer), (frame_idx, zip_ref, base_folder, gb_type, self._get_hr_gb_path(frame_idx, gb_type))))

        for gb_type, result in zip([GB_Type.BASE_COLOR, GB_Type.DEPTH, GB_Type.NORMAL], results):
            res[gb_type] = result.get()

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


    def __len__(self):
        return self.total_frames
    
    @staticmethod
    def get_datasets(settings: Settings) -> Tuple['WDSSDatasetCompressed', 'WDSSDatasetCompressed', 'WDSSDatasetCompressed']:
        """Get the training, validation, and test datasets.

        Returns:
            Tuple containing the training, validation, and test datasets.
        """
        train_dataset = WDSSDatasetCompressed(settings.train_dir, settings.frames_per_zip, settings.patch_size if settings.patched else 0, settings.upscale_factor, settings.multi_patches_per_frame, settings.num_threads)
        val_dataset = WDSSDatasetCompressed(settings.val_dir, settings.frames_per_zip, settings.patch_size if settings.patched else 0, settings.upscale_factor, settings.multi_patches_per_frame, settings.num_threads)
        test_dataset = WDSSDatasetCompressed(settings.test_dir, settings.frames_per_zip, 0, settings.upscale_factor, num_threads = settings.num_threads)

        return train_dataset, val_dataset, test_dataset
