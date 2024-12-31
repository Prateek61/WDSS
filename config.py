import torch 

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Settings:
    system_log_file = "system.log"
    system_log_to_console = True
    log_dir: str = "logs"