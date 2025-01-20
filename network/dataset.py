import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from torch.utils.data import Dataset
from enum import Enum
from utils.image_utils import ImageUtils
import torch
import zipfile
from random import randint

from typing import Dict, List

class GBufferType(Enum):
    """Enum class for GBuffer types."""
    BASE_COLOR = 0
    BASE_COLOR_AA = 1
    METALLIC = 2
    MOTION_VECTOR = 3
    NOV = 4
    POST_TONEMAP_HDR_COLOR = 5
    SCENE_DEPTH = 6
    WORLD_NORMAL = 7

class WDSSdataset(Dataset):
    "Dataset class for the WDSS dataset."
    def __init__(self, settings):
        self.settings = settings
        self.data = []
        self.high_res_path = self._get_file_paths('high_res')
        self.low_res_path = self._get_file_paths('low_res')
        self.all_g_buffer_path = self._get_file_paths('g_buffers')
        self.buffer_paths = self._group_g_buffers(60)

        print(f"Found {len(self.high_res_path)} high res images")
        print(f"Found {len(self.low_res_path)} low res images")
        print(f"Found {len(self.buffer_paths)} g buffer groups")

    def _get_file_paths(self, subfolder):
        """Retrieve file paths from a specific subfolder."""
        folder_path = os.path.join(self.settings.dataset_path, subfolder)
        return [os.path.join(folder_path, f) for f in os.listdir(folder_path)]

    def _group_g_buffers(self, group_size):
        """Group g_buffers into lists of specified group size."""
        buffer_groups = []
        num_groups = len(self.all_g_buffer_path) // group_size
        for i in range(group_size):
            buffer = [
                self.all_g_buffer_path[j] for j in range(i, len(self.all_g_buffer_path), group_size)
            ]
            buffer_groups.append(buffer)
        return buffer_groups

    def __len__(self):
        return len(self.high_res_path)
    
    def __getitem__(self, idx):
        # Load high-resolution and low-resolution images
        high_res = ImageUtils.load_exr_image_opencv(self.high_res_path[idx])
        low_res = ImageUtils.load_exr_image_opencv(self.low_res_path[idx])
                    
        # Permute dimensions to CHW (if the loaded images are in HWC format)
        high_res = high_res.transpose(2, 0, 1)  # HWC -> CHW
        low_res = low_res.transpose(2, 0, 1)    # HWC -> CHW

        # Load g_buffers and permute dimensions to CHW
        g_buffers = {
            g_buffer.name.lower(): ImageUtils.load_exr_image_opencv(self.buffer_paths[idx][g_buffer.value]).transpose(2, 0, 1)
            for g_buffer in GBufferType
        }
        
        # Create a sample dictionary
        sample = {
            'high_res': high_res,
            'low_res': low_res,
            'g_buffers': g_buffers
        }
        
        return sample
    
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

class WDSSDatasetCompressed(Dataset):
    FRAME_PATHS = {
        'SceneName': 'Asian_Village_Demo',
        'HR_FOLDER': 'High_Res',
        'LR_FOLDER': 'Low_Res',
        'HR_GB_FOLDER': 'G_Buffers',
        'LR_GB_FOLDER': 'Low_Res_G'
    }

    def __init__(self, root_dir: str, frames_per_zip: int, patch_size: int | None = None, upscale_factor: int = 2):
        self.root_dir = root_dir
        self.frames_per_zip = frames_per_zip
        self.compressed_files = os.listdir(root_dir)
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor

        self.total_frames = len(self.compressed_files) * frames_per_zip

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        zip_idx = idx // self.frames_per_zip
        frame_idx = idx % self.frames_per_zip

        zip_file = self.compressed_files[zip_idx]
        frame_idx += 1

        frames: List[torch.Tensor] = []

        with zipfile.ZipFile(os.path.join(self.root_dir, zip_file), 'r') as zip_ref:
            frame_paths = [
                self._get_hr_frame_path(frame_idx),
                self._get_lr_frame_path(frame_idx),
                self._get_hr_gb_path(frame_idx, GB_Type.BASE_COLOR),
                self._get_hr_gb_path(frame_idx, GB_Type.MOTION_VECTOR),
                self._get_hr_gb_path(frame_idx, GB_Type.DEPTH),
                self._get_hr_gb_path(frame_idx, GB_Type.NORMAL),
            ]

            for frame_path in frame_paths:
                buffer = zip_ref.read(frame_path)
                frame = ImageUtils.decode_exr_image_opencv(buffer)
                frames.append(torch.from_numpy(frame).permute(2, 0, 1))

        

        if self.patch_size is not None:
            # Get the patch position,
            # which is a random crop from the frame
            _, lr_h, lr_w = frames[1].shape
            patch_y = randint(0, lr_h - self.patch_size)
            patch_x = randint(0, lr_w - self.patch_size)
            hr_patch = frames[0][:, patch_y*self.upscale_factor:patch_y*self.upscale_factor + self.patch_size*self.upscale_factor, patch_x*self.upscale_factor:patch_x*self.upscale_factor + self.patch_size*self.upscale_factor]
            lr_patch = frames[1][:, patch_y:patch_y + self.patch_size, patch_x:patch_x + self.patch_size]
            gb_patch = torch.cat(frames[2:], dim=0)[:, patch_y*self.upscale_factor:patch_y*self.upscale_factor + self.patch_size*self.upscale_factor, patch_x*self.upscale_factor:patch_x*self.upscale_factor + self.patch_size*self.upscale_factor]
        else:
            hr_patch = frames[0]
            lr_patch = frames[1]
            gb_patch = torch.cat(frames[2:], dim=0)

        return {
            'HR': hr_patch,
            'LR': lr_patch,
            'GB': gb_patch
        }

    def _get_hr_frame_path(self, frame_idx: int) -> str:
        res = 'DATA/' + self.FRAME_PATHS['HR_FOLDER'] + '/'
        res += self.FRAME_PATHS['SceneName'] + '.' + str(frame_idx).zfill(4) + '.exr'

        return res
    
    def _get_lr_frame_path(self, frame_idx: int) -> str:
        res = 'DATA/' + self.FRAME_PATHS['LR_FOLDER'] + '/'
        res += self.FRAME_PATHS['SceneName'] + '.' + str(frame_idx).zfill(4) + '.exr'

        return res
    
    def _get_hr_gb_path(self, frame_idx: int, gb_type: GB_Type) -> str:
        res = 'DATA/' + self.FRAME_PATHS['HR_GB_FOLDER'] + '/'
        res += self.FRAME_PATHS['SceneName'] + gb_type.value + '.' + str(frame_idx).zfill(4) + '.exr'

        return res
    
    def _get_lr_gb_path(self, frame_idx: int, gb_type: GB_Type) -> str:
        res = 'DATA/' + self.FRAME_PATHS['LR_GB_FOLDER'] + '/'
        res += self.FRAME_PATHS['SceneName'] + gb_type.value + '.' + str(frame_idx).zfill(4) + '.exr'

        return res

    def __len__(self):
        return self.total_frames