import os
# To make sure opencv imports .exr files
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
# If the current directory is not WDSS, then set it to one level up
if os.getcwd()[-4:] != 'WDSS':
    os.chdir('..')
    print("Changed working dir to: " + os.getcwd())
else:
    print("Current wodking dir: " + os.getcwd())

# Imports
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as StepLR
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

# Typing 
from typing import List, Dict, Tuple, Any, Union

from config import device, Settings
from commons import initialize
from utils.image_utils import ImageUtils
from network.dataset import WDSSDataset, GB_TYPE, RawFrameGroup, FrameGroup, ZipUtils
from utils.masks import Mask
from utils.preprocessor import Preprocessor
from network.image_evaluator import ImageEvaluator
from network.trainer import Trainer
from network.models.GetModel import get_model
from utils.wavelet import WaveletProcessor
from network.losses import *
from network.model_evaluator import *

from tqdm import tqdm

def initialize_settings(config_path: str = "config/config.json", out_dir_in_config_path: bool = False, create_paths: bool = True) -> Settings:
    """
    Initialize the settings and create the output directory.
    """
    settings = Settings(settings_path=config_path, out_dir_in_config_path=out_dir_in_config_path)
    
    initialize(settings, create_paths)
    
    return settings

def get_preprocessor(settings: Settings) -> Preprocessor:
    return Preprocessor.from_config(settings.preprocessor_config)
