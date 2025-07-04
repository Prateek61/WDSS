import os

from config import Settings, device
from utils.wavelet import WaveletProcessor
from typing import TypeVar, Tuple

def initialize(settings: Settings, craete_paths: bool = True) -> None:
    """Initialize the system.

    Creates the folders for logs, models, and outputs if they do not exist.
    """
    
    # Create the folders for logs, models, and outputs if they do not exist
    if craete_paths:
        for folder in [settings.log_path(), settings.model_path()]:
            os.makedirs(folder, exist_ok=True)
    
    WaveletProcessor.WAVELET_TRANSFORM_TYPE = settings.wt_type
    WaveletProcessor.WAVELET_TYPE = settings.wavelet_type

    print(f"Job: {settings.job_name}, Device: {device}")
    print(f"Model path: {settings.model_path()}")
    print(f"Log path: {settings.log_path()}")
