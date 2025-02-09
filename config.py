import torch 
import json
import os

from typing import List, Dict, Any

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Settings:
    def __init__(self, settings_path: str = "config/config.json", model_name: str = "WDSS"):
        with open(settings_path, "r") as f:
            settings = json.load(f)
        self.job_name: str = settings["job_name"]
        self.out_dir: str = settings["out_dir"]
        self.model_dir: str = settings["model_dir"]
        self.log_dir: str = settings["log_dir"] # For tensorboard logs
        self.train_dir: str = settings["train_dir"]
        self.val_dir: str = settings["val_dir"]
        self.test_dir: str = settings["test_dir"]
        self.model_name: str = model_name
        self.frames_per_zip: int = settings["frames_per_zip"]
        self.patched: bool = settings["patched"]
        self.patch_size: int = settings["patch_size"]
        self.batch_size: int = settings["batch_size"]
        self.test_images_idx: List[int] = settings["test_images_idx"]
        self.system_log_file: str = settings["system_log_file"] # For system logs like errors, warnings, etc.
        self.model_save_interval: int = settings["model_save_interval"] # Save model every x epochs
        self.output_interval: int = settings["output_interval"] # Save output every x epochs
        self.upscale_factor: int = settings["upscale_factor"]
        self.multi_patches_per_frame: bool = settings["multi_patches_per_frame"]
        self.num_threads: int = settings['num_threads']
        self.model_config: Dict[str, Any] = settings["model_config"]
        self.preprocessor_config: Dict[str, Any] = settings["preprocessor_config"]
        self._settings = settings

    def get_full_path(self, folder: str) -> str:
        return os.path.join(self.out_dir, f'{self.job_name}-{self.model_name}', folder)
    
    def model_path(self) -> str:
        return self.get_full_path(self.model_dir)
    
    def log_path(self) -> str:
        return self.get_full_path(self.log_dir)

    def __str__(self):
        return f"Settings: {self.__dict__}"
    