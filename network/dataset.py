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

class WDSSDatasetCompressed(Dataset):
    FRAME_PATHS = {
        'HR_FOLDER': 'HighRes',
        'LR_FOLDER': 'LowRes',
        'HR_GB_FOLDER': 'HighResGBuffer',
        'LR_GB_FOLDER': 'LowResGBuffer'
    }

    def __init__(self, root_dir: str, frames_per_zip: int, patch_size: int = 0, upscale_factor: int = 2, multi_patches_per_frame: bool = False):
        self.root_dir = root_dir
        self.frames_per_zip = frames_per_zip
        self.compressed_files = os.listdir(root_dir)
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.multi_patches_per_frame = multi_patches_per_frame

        self.patches_per_frame = self._patches_per_frame((360, 640), patch_size)
        self.total_frames = len(self.compressed_files) * self.patches_per_frame * frames_per_zip


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            raw_frames = self._get_raw_frames(idx)
        except Exception as e:
            print(f'Error in getting raw frames: {e}')
            return self.__getitem__(idx)

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
        gb = torch.cat([gb, raw_frames[RawFrameGroup.HR_GB][GB_Type.SPECULAR]], dim=0)

        res[FrameGroup.GB.value] = gb

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

            def threaded_hr_gb():
                hr_gbuffers = self._get_hr_g_buffers(frame_idx, zip_ref, base_folder)
                res.update({RawFrameGroup.HR_GB: hr_gbuffers})
            def threaded_lr_gb():
                lr_gbuffers = self._get_lr_g_buffers(frame_idx, zip_ref, base_folder)
                res.update({RawFrameGroup.LR_GB: lr_gbuffers})
            def threaded_temporal_gb():
                temporal_gbuffers = self._get_temporal_g_buffers(frame_idx - 1, zip_ref, base_folder)
                res.update({RawFrameGroup.TEMPORAL_GB: temporal_gbuffers})
            def threaded_hr_frame():
                hr_frame = self._get_hr_frame(frame_idx, zip_ref, base_folder)
                res.update({RawFrameGroup.HR: hr_frame})
            def threaded_lr_frame():
                lr_frame = self._get_lr_frame(frame_idx, zip_ref, base_folder)
                res.update({RawFrameGroup.LR: lr_frame})
            def threaded_temporal_frame():
                temporal_frame = self._get_hr_frame(frame_idx - 1, zip_ref, base_folder)
                res.update({RawFrameGroup.TEMPORAL: temporal_frame})

            threads: List[threading.Thread] = []
            threads.append(threading.Thread(target=threaded_hr_gb))
            threads.append(threading.Thread(target=threaded_lr_gb))
            threads.append(threading.Thread(target=threaded_temporal_gb))
            threads.append(threading.Thread(target=threaded_hr_frame))
            threads.append(threading.Thread(target=threaded_lr_frame))
            threads.append(threading.Thread(target=threaded_temporal_frame))
            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()


        if self.patch_size and not no_patch:
            # Get the patch position
            # Which is a random crop from the frame
            _, lr_h, lr_w = res[RawFrameGroup.LR].shape
            lr_window, hr_window = self._get_random_patch_window(patch_idx ,(lr_h, lr_w), self.upscale_factor, self.patch_size)
            res[RawFrameGroup.HR] = res[RawFrameGroup.HR][:, hr_window[0][0]:hr_window[1][0], hr_window[0][1]:hr_window[1][1]]
            res[RawFrameGroup.LR] = res[RawFrameGroup.LR][:, lr_window[0][0]:lr_window[1][0], lr_window[0][1]:lr_window[1][1]]
            res[RawFrameGroup.TEMPORAL] = res[RawFrameGroup.TEMPORAL][:, hr_window[0][0]:hr_window[1][0], hr_window[0][1]:hr_window[1][1]]
            for gb_type in GB_Type:
                res[RawFrameGroup.HR_GB][gb_type] = res[RawFrameGroup.HR_GB][gb_type][:, hr_window[0][0]:hr_window[1][0], hr_window[0][1]:hr_window[1][1]]
                res[RawFrameGroup.LR_GB][gb_type] = res[RawFrameGroup.LR_GB][gb_type][:, lr_window[0][0]:lr_window[1][0], lr_window[0][1]:lr_window[1][1]]
            for gb_type in [GB_Type.BASE_COLOR, GB_Type.DEPTH, GB_Type.NORMAL]:
                res[RawFrameGroup.TEMPORAL_GB][gb_type] = res[RawFrameGroup.TEMPORAL_GB][gb_type][:, hr_window[0][0]:hr_window[1][0], hr_window[0][1]:hr_window[1][1]]

        return res


    def _get_hr_g_buffers(self, frame_idx: int, zip_ref: zipfile.ZipFile, base_folder: str) -> Dict[GB_Type, torch.Tensor]:
        res = {}

        def threaded_hr_gb(gb_type: GB_Type):
            buffer = zip_ref.read(base_folder + self._get_hr_gb_path(frame_idx, gb_type))
            frame = ImageUtils.decode_exr_image_opencv(buffer)
            if gb_type in GBufferChannels:
                frame = frame[:, :, GBufferChannels[gb_type]]
            torch_frame = torch.from_numpy(frame).permute(2, 0, 1)
            res.update({gb_type: torch_frame})

        threads: List[threading.Thread] = []
        for gb_type in GB_Type:
            threads.append(threading.Thread(target=threaded_hr_gb, args=(gb_type,)))
        for thread in threads:
            thread.start()
        for thread in threads:  
            thread.join()

        return res
    

    def _get_lr_g_buffers(self, frame_idx: int, zip_ref: zipfile.ZipFile, base_folder: str) -> Dict[GB_Type, torch.Tensor]:
        res = {}
        def threaded_lr_gb(gb_type: GB_Type):
            buffer = zip_ref.read(base_folder + self._get_lr_gb_path(frame_idx, gb_type))
            frame = ImageUtils.decode_exr_image_opencv(buffer)
            if gb_type in GBufferChannels:
                frame = frame[:, :, GBufferChannels[gb_type]]
            torch_frame = torch.from_numpy(frame).permute(2, 0, 1)
            res.update({gb_type: torch_frame})

        threads: List[threading.Thread] = []
        for gb_type in GB_Type:
            threads.append(threading.Thread(target=threaded_lr_gb, args=(gb_type,)))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        return res
    
    def _get_temporal_g_buffers(self, frame_idx: int, zip_ref: zipfile.ZipFile, base_folder: str) -> Dict[GB_Type, torch.Tensor]:
        # Just need the BaseColor, Depth, and Normal
        res = {}
        def threaded_temporal_gb(gb_type: GB_Type):
            buffer = zip_ref.read(base_folder + self._get_hr_gb_path(frame_idx, gb_type))
            frame = ImageUtils.decode_exr_image_opencv(buffer)
            if gb_type in GBufferChannels:
                frame = frame[:, :, GBufferChannels[gb_type]]
            torch_frame = torch.from_numpy(frame).permute(2, 0, 1)
            res.update({gb_type: torch_frame})

        threads: List[threading.Thread] = []

        for gb_type in [GB_Type.BASE_COLOR, GB_Type.DEPTH, GB_Type.NORMAL]:
            threads.append(threading.Thread(target=threaded_temporal_gb, args=(gb_type,)))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
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


    def _patches_per_frame(self, low_resolution: Tuple[int, int], patch_size: int) -> int:
        """Return the number of patches in the frame
        """

        if self.patch_size == 0:
            return 1
        
        if not self.multi_patches_per_frame:
            return 1
        
        if low_resolution == (360, 640) and patch_size == 256:
            return 5

        lr_h, lr_w = low_resolution
        patch_h, patch_w = patch_size, patch_size

        num_patches_h = (lr_h // patch_h) + 1
        num_patches_w = (lr_w // patch_w) + 1

        return num_patches_h * num_patches_w


    def _get_random_patch_window(self, patch_idx: int, LowResolution: Tuple[int, int], UpscaleFactor: float, PatchSize: int) -> Tuple[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Returns a patch window for the given low resolution and upscale factor and a patch index.

        Returns:
            Tuple containing the patch window (y, x) for (tl, br) in LR and HR.
        """
        if not self.multi_patches_per_frame:
            lr_h, lr_w = LowResolution
            patch_h, patch_w = PatchSize, PatchSize

            # Number of patches in the image
            num_patches_h = (lr_h // patch_h) + 1
            num_patches_w = (lr_w // patch_w) + 1

            # Randomly select a patch
            patch_idx_y = randint(0, num_patches_h - 1)
            patch_idx_x = randint(0, num_patches_w - 1)

            # Get the patch window
            patch_y = min(patch_idx_y * (lr_h // num_patches_h), lr_h - patch_h)
            patch_x = min(patch_idx_x * (lr_w // num_patches_w), lr_w - patch_w) 

            hr_patch_y_tl = patch_y * UpscaleFactor
            hr_patch_x_tl = patch_x * UpscaleFactor
            hr_patch_y_br = hr_patch_y_tl + patch_h * UpscaleFactor
            hr_patch_x_br = hr_patch_x_tl + patch_w * UpscaleFactor

            lr_window = ((patch_y, patch_x), (patch_y + patch_h, patch_x + patch_w))
            hr_window = ((hr_patch_y_tl, hr_patch_x_tl), (hr_patch_y_br, hr_patch_x_br))
            return lr_window, hr_window


        if LowResolution == (360, 640) and PatchSize == 256:
            return self._patch_window_def(patch_idx)

        # LR and patch size
        lr_h, lr_w = LowResolution
        patch_h, patch_w = PatchSize, PatchSize
        # Number of patches in hight and width direction
        num_patches_h = (lr_h // patch_h) + 1
        num_patches_w = (lr_w // patch_w) + 1
        # Patch index in hight and width direction
        patch_idx_y, patch_idx_x = divmod(patch_idx, num_patches_w)
        # Temporary stride
        tmp_stride_h = lr_h // num_patches_h
        tmp_stride_w = lr_w // num_patches_w
        # Stride in hight and width direction
        stride_h = int((lr_h - patch_h * 0.10) // num_patches_h)
        stride_w = int((lr_w - patch_w * 0.10) // num_patches_w)
        # Patch position in LR
        patch_y = int(patch_h * 0.05 + patch_idx_y * stride_h)
        patch_x = int(patch_w * 0.05 + patch_idx_x * stride_w)
        # Patch can move by 20% of the stride
        patch_y += randint(-stride_h // 5, stride_h // 5)
        patch_x += randint(-stride_w // 5, stride_w // 5)
        # Clip the patch to the frame
        patch_y = min(max(patch_y, 0), lr_h - patch_h)
        patch_x = min(max(patch_x, 0), lr_w - patch_w)

        hr_patch_y_tl = patch_y * UpscaleFactor
        hr_patch_x_tl = patch_x * UpscaleFactor
        hr_patch_y_br = hr_patch_y_tl + patch_h * UpscaleFactor
        hr_patch_x_br = hr_patch_x_tl + patch_w * UpscaleFactor

        lr_window = ((patch_y, patch_x), (patch_y + patch_h, patch_x + patch_w))
        hr_window = ((hr_patch_y_tl, hr_patch_x_tl), (hr_patch_y_br, hr_patch_x_br))
        return lr_window, hr_window

    def _patch_window_def(self, patch_idx: int) -> Tuple[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Patch window for resolution 360x640 and patch size 256 and upscale factor 2
        """

        lr_h, lr_w = 360, 640
        patch_h, patch_w = 256, 256
        num_patches_h = 2
        num_patches_w = 2
        upscale_factor = 2

        h_window = lr_h - patch_h
        w_window = lr_w - patch_w

        if patch_idx == 0:
            patch_y, patch_x = h_window * 0.1, w_window * 0.1
        elif patch_idx == 1:
            patch_y, patch_x = h_window * 0.1, (lr_w - patch_w) - w_window * 0.1
        elif patch_idx == 2:
            patch_y, patch_x = (lr_h - patch_h) - h_window * 0.1, w_window * 0.1
        elif patch_idx == 3:
            patch_y, patch_x = (lr_h - patch_h) - h_window * 0.1, (lr_w - patch_w) - w_window * 0.1
        elif patch_idx == 4:
            patch_y, patch_x = h_window // 2, w_window // 2
        else:
            raise ValueError('Invalid patch index')
        
        # Patch can move 30% of the lr_h and lr_w in the y and x direction
        patch_y += randint(-int(h_window * 0.15), int(h_window * 0.15))
        patch_x += randint(-int(w_window * 0.15), int(w_window * 0.15))

        # Clip the patch to the frame
        patch_y = int(min(max(patch_y, 0), lr_h - patch_h))
        patch_x = int(min(max(patch_x, 0), lr_w - patch_w))

        hr_patch_y_tl = patch_y * upscale_factor
        hr_patch_x_tl = patch_x * upscale_factor
        hr_patch_y_br = hr_patch_y_tl + patch_h * upscale_factor
        hr_patch_x_br = hr_patch_x_tl + patch_w * upscale_factor

        lr_window = ((patch_y, patch_x), (patch_y + patch_h, patch_x + patch_w))
        hr_window = ((hr_patch_y_tl, hr_patch_x_tl), (hr_patch_y_br, hr_patch_x_br))
        return lr_window, hr_window


    def __len__(self):
        return self.total_frames
    
    @staticmethod
    def get_datasets(settings: Settings) -> Tuple['WDSSDatasetCompressed', 'WDSSDatasetCompressed', 'WDSSDatasetCompressed']:
        """Get the training, validation, and test datasets.

        Returns:
            Tuple containing the training, validation, and test datasets.
        """
        train_dataset = WDSSDatasetCompressed(settings.train_dir, settings.frames_per_zip, settings.patch_size if settings.patched else 0, settings.upscale_factor)
        val_dataset = WDSSDatasetCompressed(settings.val_dir, settings.frames_per_zip, settings.patch_size if settings.patched else 0, settings.upscale_factor)
        test_dataset = WDSSDatasetCompressed(settings.test_dir, settings.frames_per_zip, 0, settings.upscale_factor)

        return train_dataset, val_dataset, test_dataset
