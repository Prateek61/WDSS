import torch
import json
import os

from typing import List, Dict, Any, Tuple

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Settings:
    def __init__(self, settings_path: str = "config/config.json", out_dir_in_config_path: bool = False):
        with open(settings_path, "r") as f:
            settings: Dict = json.load(f)
        self.settings_raw = settings
        self.out_dir = settings["out_dir"]
        self.model_dir: str = settings["model_dir"]
        self.log_dir: str = settings["log_dir"]  # For tensorboard logs
        if out_dir_in_config_path:
            # Get the directory path of the config file
            config_dir = os.path.dirname(settings_path)
            # Set the out_dir to the same directory as the config file
            self.out_dir = config_dir
        self.job_name: str = settings["job_name"]
        self.preprocessor_config: Dict[str, Any] = settings["preprocessor_config"]
        self.dataset_config: Dict[str, Any] = settings["dataset_config"]
        self.test_images_idx: List[Tuple[int, float]] = settings["test_images_idx"]
        self.model_save_interval: int = settings["model_save_interval"]
        self.image_log_interval: int = settings["image_log_interval"]
        self.settings_dir_path: str = os.path.dirname(settings_path)
        self.out_dir_in_config_path: bool = out_dir_in_config_path
        self.wt_type: str = settings.get("wt_type", "dwt")
        self.wavelet_type: str = settings.get("wavelet_type", "haar")
        self.decomposition_level: int = settings.get("decomposition_level", 1)

    def get_full_path(self, folder: str) -> str:
        return os.path.join(self.get_base_path(), folder)
    
    def model_path(self) -> str:
        return self.get_full_path(self.model_dir)
    
    def log_path(self) -> str:
        return self.get_full_path(self.log_dir)
    
    def get_base_path(self) -> str:
        if not self.out_dir_in_config_path:
            return os.path.join(self.out_dir, f'{self.job_name}')
        else:
            return self.settings_dir_path
    
    def save_config(self):
        with open(os.path.join(self.get_base_path(), 'config.json'), 'w') as f:
            json.dump(self.settings_raw, f, indent=4)

    # Overload the [] operator to access settings like a dictionary
    def __getitem__(self, key: str) -> Any:
        return self.settings_raw[key]

    def __str__(self):
        return f"Settings: {self.__dict__}"
