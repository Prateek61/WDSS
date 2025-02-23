import os
# Display current working directory
print(os.getcwd())
# To make sure opencv imports .exr files
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
# If the current directory is not WDSS, then set it to one level up
if os.getcwd()[-4:] != 'WDSS':
    os.chdir('..')
print(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from typing import List, Tuple, Dict

from config import device, Settings
from commons import initialize
from utils import *

settings = Settings("config/config.json", "WDSSV5")
initialize(settings=settings)

from network.dataset import *
from network.models.WDSS import get_wdss_model
from network.losses import CriterionSSIM_L1, CriterionSSIM_MSE
from network.trainer import Trainer

train_dataset, val_dataset, test_dataset = WDSSDatasetCompressed.get_datasets(settings)

def get_model() -> nn.Module:
    global settings
    return get_wdss_model(settings.model_config).to(device)

def setup_for_training() -> Trainer:
    global settings
    model = get_model()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = CriterionSSIM_MSE().to(device)
    trainer = Trainer(settings, model, optimizer, scheduler, criterion, train_dataset, val_dataset, test_dataset)

    try:
        trainer.load_best_checkpoint()
        print(f"Loaded best checkpoint, epoch {trainer.total_epochs}")
    except:
        print("No checkpoint found")

    return trainer


