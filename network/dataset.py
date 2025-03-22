import torch
from torch.utils.data import Dataset, DataLoader

import os
from zipfile import ZipFile
from random import randint
from enum import Enum
from multiprocessing.pool import ThreadPool, AsyncResult

from utils.image_utils import ImageUtils
from utils.patch import Patch
from config import Settings
from .commons import wrap_try, GB_TYPE, RawFrameGroup, FrameGroup
from utils.preprocessor import Preprocessor

from typing import Dict, List, Tuple

class ZipUtils:
    @staticmethod
    def get_frame(zip_ref: ZipFile, file_path: str) -> torch.Tensor:
        buffer = zip_ref.read(file_path)

        if not buffer:
            raise ValueError(f"Failed to read file: {file_path} from zip.")

        frame = ImageUtils.decode_exr_image_opencv(buffer)
        return torch.from_numpy(frame).permute(2, 0, 1)
    
class WDSSDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        frames_per_zip: int,
        hr_patch_size: int, # 0 for no patching
        multi_patches_per_frame: bool,
        resolutions: Dict[int, Tuple[str, Tuple[int, int]]], # Dict mapping scale factor to (folder_name, (height, width))
        multiprocessing: bool,
        preprocessor: Preprocessor
    ):
        self.root_dir = root_dir
        self.frames_per_zip = frames_per_zip
        self.hr_patch_size = hr_patch_size
        self.multi_patches_per_frame = multi_patches_per_frame
        self.resolutions = resolutions
        self.multiprocessing = multiprocessing
        self.preprocessor = preprocessor

        self._initialize()

    def _initialize(self):
        self.compressed_files = os.listdir(self.root_dir)
        self.patch = Patch(
            high_resolution=self.resolutions[1][1],
            high_resolution_patch_size=self.hr_patch_size,
            multi_patches_per_image=self.multi_patches_per_frame
        )   
        self.patches_per_frame = self.patch.patches_per_frame
        self.total_frames = len(self.compressed_files) * self.frames_per_zip
        if self.multiprocessing:
            self._thread_pool = ThreadPool(12)

    def __len__(self) -> int:
        return self.total_frames

    def get_raw_frames(self, idx: int, upscale_factor: float = 2.0, no_patch: bool = False) -> Dict[RawFrameGroup, Dict[GB_TYPE, torch.Tensor]]:
        """Get raw frames from the dataset
        """
        if self.multiprocessing:
            return self._raw_frames_parallel(idx, upscale_factor, no_patch)
        else:
            return self._raw_frames_no_parallel(idx, upscale_factor, no_patch)

    def _raw_frames_no_parallel(self, idx: int, upscale_factor: float = 2.0, no_patch: bool = False) -> Dict[RawFrameGroup, Dict[GB_TYPE, torch.Tensor]]:
        """Get raw frames from the dataset
        """
        res: Dict[RawFrameGroup, Dict[GB_TYPE, torch.Tensor]] = {
            RawFrameGroup.HR_GB: {gb_type: None for gb_type in GB_TYPE},
            RawFrameGroup.LR_GB: {gb_type: None for gb_type in GB_TYPE},
            RawFrameGroup.TEMPORAL_GB: {gb_type: None for gb_type in GB_TYPE}
        }

        if not no_patch:
            patch_idx = idx % self.patches_per_frame
            frame_no = idx // self.patches_per_frame
        else:
            frame_no = idx
            patch_idx = 1

        zip_file_idx = frame_no // self.frames_per_zip
        frame_no = frame_no % self.frames_per_zip
        frame_no += 3 # Ignore the first three frames

        if self.hr_patch_size and not no_patch:
            lr_window, hr_window = self.patch.get_patch_window(upscale_factor, patch_idx)
        else:
            lr_window, hr_window = None, None

        # Open the zip file
        with ZipFile(os.path.join(self.root_dir, self.compressed_files[zip_file_idx]), 'r') as zip_ref:
            base_folder = self._get_base_folder_name(zip_ref)

            for gb_type in GB_TYPE:
                res[RawFrameGroup.HR_GB][gb_type] = self._process_frame(zip_ref, self._get_path(base_folder, frame_no, gb_type, 1), hr_window)
                res[RawFrameGroup.LR_GB][gb_type] = self._process_frame(zip_ref, self._get_path(base_folder, frame_no, gb_type, upscale_factor), lr_window)
                res[RawFrameGroup.TEMPORAL_GB][gb_type] = self._process_frame(zip_ref, self._get_path(base_folder, frame_no - 1, gb_type, 1), hr_window)

        return res 

    def _raw_frames_parallel(self, idx: int, upscale_factor: float = 2.0, no_patch: bool = False) -> Dict[RawFrameGroup, Dict[GB_TYPE, torch.Tensor]]:
        """Get raw frames from the dataset
        """
        res: Dict[RawFrameGroup, Dict[GB_TYPE, torch.Tensor]] = {
            RawFrameGroup.HR_GB: {gb_type: None for gb_type in GB_TYPE},
            RawFrameGroup.LR_GB: {gb_type: None for gb_type in GB_TYPE},
            RawFrameGroup.TEMPORAL_GB: {gb_type: None for gb_type in GB_TYPE}
        }

        if not no_patch:
            patch_idx = idx % self.patches_per_frame
            frame_no = idx // self.patches_per_frame
        else:
            frame_no = idx
            patch_idx = 1
        zip_file_idx = frame_no // self.frames_per_zip
        frame_no = frame_no % self.frames_per_zip
        frame_no += 3 # Ignore the first three frames

        if self.hr_patch_size and not no_patch:
            lr_window, hr_window = self.patch.get_patch_window(upscale_factor, patch_idx)
        else:
            lr_window, hr_window = None, None

        # Open the zip file
        with ZipFile(os.path.join(self.root_dir, self.compressed_files[zip_file_idx]), 'r') as zip_ref:
            base_folder = self._get_base_folder_name(zip_ref)

            # Async results
            hr_async_res: Dict[GB_TYPE, AsyncResult] = {}
            lr_async_res: Dict[GB_TYPE, AsyncResult] = {}
            temporal_async_res: Dict[GB_TYPE, AsyncResult] = {}

            for gb_type in GB_TYPE:
                hr_async_res[gb_type] = self._thread_pool.apply_async(
                    WDSSDataset._process_frame,
                    (zip_ref, self._get_path(base_folder, frame_no, gb_type, 1), hr_window)
                )
                lr_async_res[gb_type] = self._thread_pool.apply_async(
                    WDSSDataset._process_frame,
                    (zip_ref, self._get_path(base_folder, frame_no, gb_type, upscale_factor), lr_window)
                )
                temporal_async_res[gb_type] = self._thread_pool.apply_async(
                    WDSSDataset._process_frame,
                    (zip_ref, self._get_path(base_folder, frame_no - 1, gb_type, 1), hr_window)
                )

            # Wait for the async results
            for gb_type in GB_TYPE:
                res[RawFrameGroup.HR_GB][gb_type] = hr_async_res[gb_type].get()
                res[RawFrameGroup.LR_GB][gb_type] = lr_async_res[gb_type].get()
                res[RawFrameGroup.TEMPORAL_GB][gb_type] = temporal_async_res[gb_type].get()

        return res

    @staticmethod
    def _process_frame(zip_ref: ZipFile, file_path: str, patch_window: Tuple[Tuple[int, int], Tuple[int, int]]) -> torch.Tensor:
        frame = ZipUtils.get_frame(zip_ref, file_path)
        if patch_window:
            frame = WDSSDataset._apply_patch(frame, patch_window)
        return frame

    @staticmethod
    def _apply_patch(frame: torch.Tensor, patch: Tuple[Tuple[int, int], Tuple[int, int]]) -> Dict[GB_TYPE, torch.Tensor]:
        """Apply patch to the frames
        """
        frame = frame[:, patch[0][0]:patch[1][0], patch[0][1]:patch[1][1]]
        return frame

    def _get_base_folder_name(self, zip_file: ZipFile) -> str:
        """Get the base folder name from the zip file
        """
        return zip_file.namelist()[0].split('/')[0]
    
    def _get_path(self, base_folder: str, frame_no: int, gb_type: GB_TYPE, scale_factor: float):
        ext = '.' + str(frame_no).zfill(4) + '.exr'
        folder = base_folder + '/' + self.resolutions[scale_factor][0] + '/'
        file_name = gb_type.value + ext
        return folder + file_name
    