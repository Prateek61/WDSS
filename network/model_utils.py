import torch
import torch.nn as nn
import torch.optim as optim

from .losses import CriterionBase
from .models.ModelBase import ModelBase

from typing import Dict, Tuple

class ModelUtils:
    @staticmethod
    def save_checkpoint(model: ModelBase, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler, step: int, validation_loss: float, checkpoint_path: str):
        """Save the model checkpoint
        """
        checkpoint = {
            'step': step,
            'validation_loss': validation_loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)

    @staticmethod
    def load_checkpoint(model: ModelBase, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler, checkpoint_path: str) -> Tuple[int, float]:
        """Load the model checkpoint
        """

        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['step']
        validation_loss = checkpoint['validation_loss']
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except:
            pass
        return step, validation_loss
