import config
import logging
import torch

from typing import TypeVar, Tuple

def initialize(settings: config.Settings = config.Settings()):
    initialize_system_log(settings.system_log_file)
    logging.info(f"Device: {config.device}")

def initialize_system_log(log_file: str = "") -> None:
    # Logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Basic configuration
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Enable logging to file
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logging.getLogger('').addHandler(file_handler)

    logging.info("System Logger initialized")
    