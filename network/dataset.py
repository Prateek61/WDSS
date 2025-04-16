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
    
    @wrap_try
    def __getitem__(self, index) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        raw_frames = self.get_raw_frames(index)
        return self.preprocessor.preprocess(raw_frames, 2.0)
    
    @wrap_try
    def get_item(self, index, upscale_factor: float, no_patch: bool = False) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        """Get item from the dataset
        """
        raw_frames = self.get_raw_frames(index, upscale_factor, no_patch)
        return self.preprocessor.preprocess(raw_frames, upscale_factor)

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
    
    @staticmethod
    def get_datasets(
        settings: Settings
    ) -> Tuple['WDSSDataset', 'WDSSDataset', 'WDSSDataset']:
        """Get the datasets for training, validation and testing
        """

        preprocessor = Preprocessor.from_config(settings.preprocessor_config)
        train_dir = settings.get_full_path(settings.dataset_config['train_dir'])
        val_dir = settings.get_full_path(settings.dataset_config['val_dir'])
        test_dir = settings.get_full_path(settings.dataset_config['test_dir'])
        frames_per_zip = settings.dataset_config['frames_per_zip']
        hr_patch_size = settings.dataset_config['patch_size']
        multi_patches_per_frame = settings.dataset_config['multi_patches_per_frame']
        multiprocessing = settings.dataset_config['multiprocessing']
        resolutions = {}
        for key, value in settings.dataset_config['resolutions'].items():
            resolutions[int(key)] = (value["folder"], (value["height"], value["width"]))
        
        train_dataset = WDSSDataset(
            root_dir=train_dir,
            frames_per_zip=frames_per_zip,
            hr_patch_size=hr_patch_size,
            multi_patches_per_frame=multi_patches_per_frame,
            resolutions=resolutions,
            multiprocessing=multiprocessing,
            preprocessor=preprocessor
        )
        val_dataset = WDSSDataset(
            root_dir=val_dir,
            frames_per_zip=frames_per_zip,
            hr_patch_size=hr_patch_size,
            multi_patches_per_frame=multi_patches_per_frame,
            resolutions=resolutions,
            multiprocessing=multiprocessing,
            preprocessor=preprocessor
        )
        test_dataset = WDSSDataset(
            root_dir=test_dir,
            frames_per_zip=frames_per_zip,
            hr_patch_size=hr_patch_size,
            multi_patches_per_frame=multi_patches_per_frame,
            resolutions=resolutions,
            multiprocessing=multiprocessing,
            preprocessor=preprocessor
        )

        return train_dataset, val_dataset, test_dataset
    
class WDSSDataLoader(DataLoader):
    """Override the default DataLoader as we need to pass upscale factor to the dataset
    """
    def __init__(
        self,
        dataset: WDSSDataset,
        upscale_factors: List[Tuple[int, float]], # Upscale factors with their probabilities
        *args,
        **kwargs
    ):
        super().__init__(dataset, *args, **kwargs)
        self.upscale_factors = upscale_factors

    def __iter__(self):
        # Select a random upscale factor for each batch
        upscale_factor = self._select_random_upscale_factor()
        self.dataset.__getitem__ = lambda idx: self.dataset.get_item(idx, upscale_factor, no_patch=False)
        return super().__iter__()
    
    def _select_random_upscale_factor(self) -> float:
        """Select a random upscale factor based on the probabilities
        """
        total = sum(prob for _, prob in self.upscale_factors)
        rand_val = randint(0, total - 1)
        cumulative_prob = 0
        for factor, prob in self.upscale_factors:
            cumulative_prob += prob
            if rand_val < cumulative_prob:
                return factor
        return self.upscale_factors[-1][0]
