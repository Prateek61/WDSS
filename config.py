import torch 

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Settings:
    system_log_file = "system.log"
    system_log_to_console = True
    log_dir: str = "logs"
    dataset_path: str = "D:\Datasets\DataV1"
    system_load_model: bool = False
    system_model_path: str = "models"
    system_model_name: str = "model.pth"
    system_model_save_interval: int = 1
    