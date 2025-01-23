import torch 
import json

from typing import List, Dict

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class Settings:
#     system_log_file = "system.log"
#     system_log_to_console = True
#     log_dir: str = "logs"
#     dataset_path = r"D:\Datasets\DataV2\Data"
#     system_load_model: bool = False
#     system_model_path: str = "models"
#     system_model_name: str = "model.pth"
#     system_model_save_interval: int = 1

class Settings:
    def __init__(self, settings_path: str = "config/config.json", model_name: str = "WDSS"):
        with open(settings_path, "r") as f:
            settings = json.load(f)
        self.job_name: str = settings["job_name"]
        self.model_dir: str = settings["model_dir"]
        self.log_dir: str = settings["log_dir"] # For tensorboard logs
        self.out_dir: str = settings["out_dir"]
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

    def __str__(self):
        return f"Settings: {self.__dict__}"
    