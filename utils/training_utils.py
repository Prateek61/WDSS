import os
import torch
import tensorboard


class TrainingUtils:
    
    def __init__(self , log_dir , checkpoint_dir):
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.writer = tensorboard.SummaryWriter(log_dir)
    
    @staticmethod
    def save_checkpoint(model, optimizer, step, checkpoint_dir):
        """
        Save model checkpoint.
        """
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, checkpoint_dir)
    
    @staticmethod
    def load_checkpoint(model, optimizer, checkpoint_dir):
        """
        Load model checkpoint.
        """
        checkpoint = torch.load(checkpoint_dir)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['step']
        return step
