import os

from config import Settings, device

from typing import TypeVar, Tuple

def initialize(settings: Settings) -> None:
    """Initialize the system.

    Creates the folders for logs, models, and outputs if they do not exist.
    """
    
    # Create the folders for logs, models, and outputs if they do not exist
    for folder in [settings.log_path(), settings.model_path(), settings.out_path()]:
        os.makedirs(folder, exist_ok=True)
    
    print(f"Job: {settings.job_name}, Model: {settings.model_name}, Device: {device}")
