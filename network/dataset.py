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
from config import device

from typing import Dict, List, Tuple

class ZipUtils:
    """Utility class to handle zip files
    """
    dynamic_extraction: bool = False
    root_dir: str = None

    @staticmethod
    @wrap_try
    def get_frame(zip_ref: ZipFile, file_path: str) -> torch.Tensor:
        """Get the frame from the zip file
        """
        real_file_path = os.path.join(ZipUtils.root_dir, zip_ref.filename.split('.')[0], file_path)
        exists: bool = os.path.exists(real_file_path) and ZipUtils.dynamic_extraction
        
        if exists:
            # Read the file from disk
            with open(real_file_path, 'rb') as f:
                buffer = f.read()
        else:
            buffer = zip_ref.read(file_path)

        if not buffer:
            raise ValueError(f"Failed to read file: {file_path} from zip.")

        frame = ImageUtils.decode_exr_image_opencv(buffer)

        if not exists and ZipUtils.dynamic_extraction:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(real_file_path), exist_ok=True)
            # Write the file to disk
            with open(real_file_path, 'wb') as f:
                f.write(buffer)

        return torch.from_numpy(frame).permute(2, 0, 1)
    
PRETONEMAP_MAX_VAL: float = 10.0
DEPTH_MAX_VAL: float = 100.0

class WDSSDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        frames_per_zip: int,
        hr_patch_size: int, # 0 for no patching
        multi_patches_per_frame: bool,
        resolutions: Dict[float, Tuple[str, Tuple[int, int]]], # Dict mapping scale factor to (folder_name, (height, width))
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
        ZipUtils.root_dir = root_dir
        self.default_upscale_factor = 2.0
        self.upscale_factors = [k for k in resolutions.keys() if k != 1.0]

        self._initialize()

    def _initialize(self):
        self.compressed_files = [f for f in os.listdir(self.root_dir) if f.endswith('.zip')]
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
        raw_frames = self.get_raw_frames(index, self.default_upscale_factor)
        return self.preprocessor.preprocess(raw_frames)
    
    # @wrap_try
    def get_item(self, index, upscale_factor: float, no_patch: bool = False) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        """Get item from the dataset
        """
        raw_frames = self.get_raw_frames(index, upscale_factor, no_patch)
        return self.preprocessor.preprocess(raw_frames)
    
    # @wrap_try
    def get_log_frame(self, index: int, upscale_factor: float = 2.0, no_patch: bool = False) -> torch.Tensor:
        """Get log frame from the dataset
        """
        raw_frames = self.get_raw_frames(index, upscale_factor, no_patch)
        return self.preprocessor.get_log(raw_frames)

    def get_raw_frames(self, idx: int, upscale_factor: float = 2.0, no_patch: bool = False) -> Dict[RawFrameGroup, Dict[GB_TYPE, torch.Tensor]]:
        """Get raw frames from the dataset
        """
        if self.multiprocessing:
            return self._raw_frames_parallel(idx, upscale_factor, no_patch)
        else:
            return self._raw_frames_no_parallel(idx, upscale_factor, no_patch)
        
    def get_random_upscale_factor(self) -> float:
        """Get a random upscale factor from the list of available upscale factors
        """
        return self.upscale_factors[randint(0, len(self.upscale_factors) - 1)]
    
    def set_random_upscale_factor(self) -> None:
        """Set a random upscale factor from the list of available upscale factors
        """
        self.default_upscale_factor = self.get_random_upscale_factor()

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

            # Clamp the pretonemap values to [0, 64]
            res[RawFrameGroup.HR_GB][GB_TYPE.PRETONEMAP_METALLIC][0:3, :, :] = torch.clamp(res[RawFrameGroup.HR_GB][GB_TYPE.PRETONEMAP_METALLIC][0:3, :, :], 0.0, PRETONEMAP_MAX_VAL)
            res[RawFrameGroup.LR_GB][GB_TYPE.PRETONEMAP_METALLIC][0:3, :, :] = torch.clamp(res[RawFrameGroup.LR_GB][GB_TYPE.PRETONEMAP_METALLIC][0:3, :, :], 0.0, PRETONEMAP_MAX_VAL)
            res[RawFrameGroup.TEMPORAL_GB][GB_TYPE.PRETONEMAP_METALLIC][0:3, :, :] = torch.clamp(res[RawFrameGroup.TEMPORAL_GB][GB_TYPE.PRETONEMAP_METALLIC][0:3, :, :], 0.0, PRETONEMAP_MAX_VAL)
            # res[RawFrameGroup.HR_GB][GB_TYPE.BASE_COLOR_DEPTH][3:4, :, :] = torch.clamp(res[RawFrameGroup.HR_GB][GB_TYPE.BASE_COLOR_DEPTH][3:4, :, :], 0.0, DEPTH_MAX_VAL)
            # res[RawFrameGroup.LR_GB][GB_TYPE.BASE_COLOR_DEPTH][3:4, :, :] = torch.clamp(res[RawFrameGroup.LR_GB][GB_TYPE.BASE_COLOR_DEPTH][3:4, :, :], 0.0, DEPTH_MAX_VAL)
            # res[RawFrameGroup.TEMPORAL_GB][GB_TYPE.BASE_COLOR_DEPTH][3:4, :, :] = torch.clamp(res[RawFrameGroup.TEMPORAL_GB][GB_TYPE.BASE_COLOR_DEPTH][3:4, :, :], 0.0, DEPTH_MAX_VAL)
            res[RawFrameGroup.HR_GB][GB_TYPE.BASE_COLOR_DEPTH][3:4, :, :] = torch.where(
                res[RawFrameGroup.HR_GB][GB_TYPE.BASE_COLOR_DEPTH][3:4, :, :] > DEPTH_MAX_VAL,
                torch.tensor(0, device="cpu"),
                res[RawFrameGroup.HR_GB][GB_TYPE.BASE_COLOR_DEPTH][3:4, :, :]
            )
            res[RawFrameGroup.LR_GB][GB_TYPE.BASE_COLOR_DEPTH][3:4, :, :] = torch.where(
                res[RawFrameGroup.LR_GB][GB_TYPE.BASE_COLOR_DEPTH][3:4, :, :] > DEPTH_MAX_VAL,
                torch.tensor(0, device="cpu"),
                res[RawFrameGroup.LR_GB][GB_TYPE.BASE_COLOR_DEPTH][3:4, :, :]
            )
            res[RawFrameGroup.TEMPORAL_GB][GB_TYPE.BASE_COLOR_DEPTH][3:4, :, :] = torch.where(
                res[RawFrameGroup.TEMPORAL_GB][GB_TYPE.BASE_COLOR_DEPTH][3:4, :, :] > DEPTH_MAX_VAL,
                torch.tensor(0, device="cpu"),
                res[RawFrameGroup.TEMPORAL_GB][GB_TYPE.BASE_COLOR_DEPTH][3:4, :, :]
            )

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
    def batch_to_device(
        batch: Dict[str, torch.Tensor | Dict[str, torch.Tensor]],
        device: torch.device = device
    ):
        """Move the batch to the device
        """
        for key, value in batch.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        batch[key][sub_key] = sub_value.to(device)
            else:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = value.to(device)

        return batch
    
    @staticmethod
    def unsqueeze_batch(
        batch: Dict[str, torch.Tensor | Dict[str, torch.Tensor]],
        dim: int = 0
    ) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        """Unsqueeze the batch
        """
        for key, value in batch.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        batch[key][sub_key] = sub_value.unsqueeze(dim)
            else:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = value.unsqueeze(dim)

        return batch

    @staticmethod
    def get_datasets(
        settings: Settings
    ) -> Tuple['WDSSDataset', 'WDSSDataset', 'WDSSDataset']:
        """Get the datasets for training, validation and testing
        """

        preprocessor = Preprocessor.from_config(settings.preprocessor_config)
        train_dir = settings.dataset_config['train_dir']
        val_dir = settings.dataset_config['val_dir']
        test_dir = settings.dataset_config['test_dir']
        frames_per_zip_train = settings.dataset_config['frames_per_zip_train']
        frames_per_zip_val = settings.dataset_config['frames_per_zip_val']
        frames_per_zip_test = settings.dataset_config['frames_per_zip_test']
        hr_patch_size = settings.dataset_config['patch_size']
        multi_patches_per_frame = settings.dataset_config['multi_patches_per_frame']
        multiprocessing = settings.dataset_config['multiprocessing']
        resolutions = {}
        for key, value in settings.dataset_config['resolutions'].items():
            resolutions[float(key)] = (value["folder"], (value["resolution"][1], value["resolution"][0]))
        
        train_dataset = WDSSDataset(
            root_dir=train_dir,
            frames_per_zip=frames_per_zip_train,
            hr_patch_size=hr_patch_size,
            multi_patches_per_frame=multi_patches_per_frame,
            resolutions=resolutions,
            multiprocessing=multiprocessing,
            preprocessor=preprocessor
        )
        val_dataset = WDSSDataset(
            root_dir=val_dir,
            frames_per_zip=frames_per_zip_val,
            hr_patch_size=hr_patch_size,
            multi_patches_per_frame=multi_patches_per_frame,
            resolutions=resolutions,
            multiprocessing=multiprocessing,
            preprocessor=preprocessor
        )
        test_dataset = WDSSDataset(
            root_dir=test_dir,
            frames_per_zip=frames_per_zip_test,
            hr_patch_size=hr_patch_size,
            multi_patches_per_frame=multi_patches_per_frame,
            resolutions=resolutions,
            multiprocessing=multiprocessing,
            preprocessor=preprocessor
        )

        return train_dataset, val_dataset, test_dataset
