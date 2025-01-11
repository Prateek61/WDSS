import torch
from torch.utils.data import Dataset, DataLoader
from config import device
import config
import os
import numpy as np

class WDSSdataset(Dataset):
    def __init__(self, settings: config.Settings):
        self.settings = settings
        self.data = []
        self.settings = settings
        self.high_res_path += [os.path.join(settings.data_path, "High_res")]
        
        print("High_res_path: ", self.high_res_path)



if __name__ == "__main__":
    settings = config.Settings()
    dataset = WDSSdataset(settings)
    print("Dataset created")
        