import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import tqdm

from typing import List, Tuple, Dict

from config import device, Settings
from commons import initialize

settings = Settings("config/config.json")
initialize(settings)

from network.models.WDSS import get_wdss_model

model = get_wdss_model(settings.model_config).to(device=device)

from network.dataset import *
settings.model_name = "WDSS"
settings.num_threads = 8
train, val, test = WDSSDatasetCompressed.get_datasets(settings)

frame = test[0]

lr = frame['LR'].unsqueeze(0).to(device)
gb = frame['GB'].unsqueeze(0).to(device)
temporal = frame['TEMPORAL'].unsqueeze(0).to(device)

model.eval()

for i in range(10):
    with torch.no_grad():
        out = model(lr, gb, temporal, 2.0)