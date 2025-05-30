import concurrent.futures
import os
from .image_utils import ImageUtils

import concurrent
from tqdm import tqdm
from enum import Enum
import numpy as np
import cv2
import Imath

from typing import List, Tuple

class GBufferPacked(Enum):
    NORMAL_SPECULAR = "NormalSpecular"
    BASE_COLOR_DEPTH = "BaseColorDepth"
    PRETONEMAP_METALLIC = "PretonemapMetallic"
    MV_ROUGHNESS_NOV = 'MV_Roughness_NOV'

class RawGBuffer(Enum):
    BASE_COLOR = "FinalImageBaseColor"
    METALLIC = "FinalImageMetallic"
    #MOTION_VECTOR = "FinalImageMovieRenderQueue_MotionVectors"
    MOTION_VECTOR = "FinalImageMotionVector"
    PRE_TONEMAP = "FinalImagePreTonemapHDRColor"
    ROUGHNESS = "FinalImageRoughness"
    DEPTH = "FinalImageSceneDepth"
    SPECULAR = "FinalImageSpecular"
    NORMAL = "FinalImageWorldNormal"
    NoV = "FinalImageNoV"

class Config:
    GBufferPackedClass = GBufferPacked
    RawGBufferClass = RawGBuffer
    Delete = True
    Scale_Metallic = 1.0
    Scale_Specular = 1.0
    TargetSizes = [
        ("1080P", (1920, 1080)),
        ("540P", (960, 540)),
        ("360P", (640, 360)),
        ("270P", (480, 270))
    ]
    Train_Data_Range = (2, 188)
    Val_Data_Range = (2, 103)

class FileUtils:
    @staticmethod
    def get_full_path(base_path: str, gb_name: str, frame_no: int) -> str:
        return base_path + os.path.sep + gb_name + "." + str(frame_no).zfill(4) + ".exr"

    @staticmethod
    def load(base_path: str, gb_name: str, frame_no: int) -> np.ndarray:
        full_path = FileUtils.get_full_path(base_path, gb_name, frame_no)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File {full_path} does not exist.")
        return ImageUtils.load_exr_image_opencv(full_path)

    @staticmethod
    def save(base_path: str, gb_name: str, frame_no: int, image: np.ndarray, precisions: List[Imath.PixelType] = None) -> None:
        full_path = FileUtils.get_full_path(base_path, gb_name, frame_no)
        ImageUtils.save_exr_image_openexr(image, full_path, compression=True, precisions=precisions)
        # ImageUtils.save_exr_image_opencv(image, full_path)

class Pack:
    @staticmethod
    def pack_normal_specular(path: str, frame_no: int):
        normal = FileUtils.load(path, Config.RawGBufferClass.NORMAL.value, frame_no)[:, :, :3]
        specular = FileUtils.load(path, Config.RawGBufferClass.SPECULAR.value, frame_no)[:, :, :1]

        if Config.Scale_Specular != 1.0:
            specular = specular * Config.Scale_Specular

        packed = np.concatenate((normal, specular), axis=2)
        FileUtils.save(path, Config.GBufferPackedClass.NORMAL_SPECULAR.value, frame_no, packed,
            precisions=[Imath.PixelType.FLOAT, Imath.PixelType.FLOAT, Imath.PixelType.HALF, Imath.PixelType.HALF]          
        )

        if Config.Delete:
            os.remove(FileUtils.get_full_path(path, Config.RawGBufferClass.NORMAL.value, frame_no))
            os.remove(FileUtils.get_full_path(path, Config.RawGBufferClass.SPECULAR.value, frame_no))

    @staticmethod
    def pack_base_color_depth(path: str, frame_no: int):
        base_color = FileUtils.load(path, Config.RawGBufferClass.BASE_COLOR.value, frame_no)[:, :, :3]
        depth = FileUtils.load(path, Config.RawGBufferClass.DEPTH.value, frame_no)[:, :, :1]

        packed = np.concatenate((base_color, depth), axis=2)
        FileUtils.save(path, 
            Config.GBufferPackedClass.BASE_COLOR_DEPTH.value, 
            frame_no, packed,
            precisions=[Imath.PixelType.HALF, Imath.PixelType.HALF, Imath.PixelType.HALF, Imath.PixelType.FLOAT]
        )

        if Config.Delete:
            os.remove(FileUtils.get_full_path(path, Config.RawGBufferClass.BASE_COLOR.value, frame_no))
            os.remove(FileUtils.get_full_path(path, Config.RawGBufferClass.DEPTH.value, frame_no))

    @staticmethod
    def pack_pre_tonemap_metallic(path: str, frame_no: int):
        pre_tonemap = FileUtils.load(path, Config.RawGBufferClass.PRE_TONEMAP.value, frame_no)[:, :, :3]

        # Clamp to range [0, 64] for hdr images
        pre_tonemap = np.clip(pre_tonemap, 0.0, 64.0)

        metallic = FileUtils.load(path, Config.RawGBufferClass.METALLIC.value, frame_no)[:, :, :1]

        if Config.Scale_Metallic != 1.0:
            metallic = metallic * Config.Scale_Metallic

        packed = np.concatenate((pre_tonemap, metallic), axis=2)
        FileUtils.save(path, 
            Config.GBufferPackedClass.PRETONEMAP_METALLIC.value, 
            frame_no, 
            packed,
            precisions=[Imath.PixelType.FLOAT, Imath.PixelType.FLOAT, Imath.PixelType.FLOAT, Imath.PixelType.HALF]
        )

        if Config.Delete:
            os.remove(FileUtils.get_full_path(path, Config.RawGBufferClass.PRE_TONEMAP.value, frame_no))
            os.remove(FileUtils.get_full_path(path, Config.RawGBufferClass.METALLIC.value, frame_no))

    @staticmethod
    def _process_mv(mv: np.ndarray) -> np.ndarray:
        # Pixle space motion vector from unreal's velocity buffer
        h, w, _ = mv.shape
        mv = (mv - 0.5) * 2.0
        mv[:, :, 0] = mv[:, :, 0] * float(w)
        mv[:, :, 1] = mv[:, :, 1] * float(h)
        return mv

    @staticmethod
    def pack_mv_roughness_nov(path: str, frame_no: int):
        mv = FileUtils.load(path, Config.RawGBufferClass.MOTION_VECTOR.value, frame_no)[:, :, :2]
        roughness = FileUtils.load(path, Config.RawGBufferClass.ROUGHNESS.value, frame_no)[:, :, :1]
        nov = FileUtils.load(path, Config.RawGBufferClass.NoV.value, frame_no)[:, :, :1]

        #mv = Pack._process_mv(mv)

        packed = np.concatenate((mv, roughness, nov), axis=2)
        FileUtils.save(path, GBufferPacked.MV_ROUGHNESS_NOV.value, frame_no, packed,
            precisions=[Imath.PixelType.FLOAT, Imath.PixelType.FLOAT, Imath.PixelType.HALF, Imath.PixelType.FLOAT]
        )

        if Config.Delete:
            os.remove(FileUtils.get_full_path(path, Config.RawGBufferClass.MOTION_VECTOR.value, frame_no))
            os.remove(FileUtils.get_full_path(path, Config.RawGBufferClass.ROUGHNESS.value, frame_no))
            os.remove(FileUtils.get_full_path(path, Config.RawGBufferClass.NoV.value, frame_no))

    @staticmethod
    def pack_single_frame(path: str, frame_no: int):
        Pack.pack_normal_specular(path, frame_no)
        Pack.pack_base_color_depth(path, frame_no)
        Pack.pack_pre_tonemap_metallic(path, frame_no)
        Pack.pack_mv_roughness_nov(path, frame_no)

    @staticmethod
    def pack_all(path: str, frame_range: Tuple[int, int]) -> None:
        def pack_frame(frame_no: int) -> None:
            Pack.pack_single_frame(path, frame_no)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(pack_frame, range(frame_range[0], frame_range[1] + 1)), total=frame_range[1] - frame_range[0] + 1, desc="Packing frames", unit="frame"))

class Downsample:
    @staticmethod
    def downsample_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST_EXACT)
    
    @staticmethod
    def downsample_and_save_single_frame(source_path: str, target_path: str, frame_no: int, target_sizes: List[Tuple[str, Tuple[int, int]]] = Config.TargetSizes) -> None:
        for gb_type in Config.GBufferPackedClass:
            iamge = FileUtils.load(source_path, gb_type.value, frame_no)
            for target_name, target_size in target_sizes:
                downsampled_image = Downsample.downsample_image(iamge, target_size)

                if gb_type == GBufferPacked.PRETONEMAP_METALLIC:
                    downsampled_image[:, :, 0:3] = np.clip(downsampled_image[:, :, 0:3], 0.0, 64.0)

                FileUtils.save(target_path + os.path.sep + target_name, gb_type.value, frame_no, downsampled_image)
            
            if Config.Delete:
                os.remove(FileUtils.get_full_path(source_path, gb_type.value, frame_no))

    @staticmethod
    def make_paths(target_path: str) -> None:
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        for target_name, _ in Config.TargetSizes:
            target_dir = os.path.join(target_path, target_name)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

    @staticmethod
    def downsample_all(source_path: str, target_path: str, frame_range: Tuple[int, int]) -> None:
        Downsample.make_paths(target_path)

        def downsample_frame(frame_no: int) -> None:
            Downsample.downsample_and_save_single_frame(source_path, target_path, frame_no, Config.TargetSizes)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(downsample_frame, range(frame_range[0], frame_range[1] + 1)), total=frame_range[1] - frame_range[0] + 1, desc="Downsampling frames", unit="frame"))
            
def pack_and_downsample(source_path: str, target_path: str, frame_range: Tuple[int, int]) -> None:
    Pack.pack_all(source_path, frame_range)
    Downsample.downsample_all(source_path, target_path, frame_range)
    