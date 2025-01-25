import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import threading

from .model_utils import ModelUtils
from .losses import CriterionBase
from .models.ModelBase import ModelBase
from .dataset import *
from utils.wdss_logger import NetworkLogger
from utils.wavelet import WaveletProcessor
from config import device, Settings

from typing import Dict, Tuple

class Trainer:
    def __init__(self, settings: Settings, 
                 model: ModelBase, 
                 optimizer: torch.optim.Optimizer, 
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 criterion: CriterionBase,
                 train_dataset: WDSSDatasetCompressed,
                 validation_dataset: WDSSDatasetCompressed,
                 test_dataset: WDSSDatasetCompressed):
        super(Trainer, self).__init__()

        self.settings = settings
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.best_validation_loss = float('inf')
        self.total_epochs = 0
        self.logger = NetworkLogger(settings.log_path())
        self.test_dataset = test_dataset

        # For threading
        self._batch_loss: float | None = None
        self._batch_losses_all: Dict[str, float] = {}

    def _train_batch(self, batch: Dict[str, torch.Tensor] = {}) -> None:
        """Trains the model on a single batch of data. Runs in parallel core while loading the next batch in the main core.
        """
        # If the batch is empty, return
        if not batch:
            return

        # Move the batch to the device
        for key in batch:
            batch[key] = batch[key].to(device)

        lr_inp = batch[FrameGroup.LR.value]
        gb_inp = batch[FrameGroup.GB.value]
        temporal_inp = batch[FrameGroup.TEMPORAL.value]

        hr_gt = batch[FrameGroup.HR.value]
        hr_wavelet = WaveletProcessor.batch_wt(hr_gt)

        # Zero the gradients
        self.optimizer.zero_grad()

        # Forward pass
        wavelet, img = self.model.forward(lr_inp, gb_inp, temporal_inp)

        # Calculate the loss
        total_loss, losses = self.criterion.forward(wavelet, hr_wavelet, img, hr_gt)

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        losses_float: Dict[str, float] = {key: value.item() for key, value in losses.items()}
        
        # Update the batch loss
        self._batch_loss = total_loss.item()
        self._batch_losses_all = losses_float


    def _validate_batch(self, batch: Dict[str, torch.Tensor] = {}) -> None:
        """Validates the model on a single batch of data. Runs in parallel core while loading the next batch in the main core.
        """
        # If the batch is empty, return
        if not batch:
            return 

        # Move the batch to the device
        for key in batch:
            batch[key] = batch[key].to(device)

        lr_inp = batch[FrameGroup.LR.value]
        gb_inp = batch[FrameGroup.GB.value]
        temporal_inp = batch[FrameGroup.TEMPORAL.value]

        hr_gt = batch[FrameGroup.HR.value]
        hr_wavelet = WaveletProcessor.batch_wt(hr_gt)

        # Forward pass
        with torch.no_grad():
            wavelet, img = self.model.forward(lr_inp, gb_inp, temporal_inp)

            # Calculate the loss
            total_loss, losses = self.criterion.forward(wavelet, hr_wavelet, img, hr_gt)

        losses_float: Dict[str, float] = {key: value.item() for key, value in losses.items()}

        # Update the batch loss
        self._batch_loss = total_loss.item()
        self._batch_losses_all = losses_float


    def log_test_images(self, step: int):
        """Log the test images to tensorboard.
        """

        for i in self.settings.test_images_idx:
            frame = self.test_dataset[i]

            lr_inp = frame[FrameGroup.LR.value].unsqueeze(0).to(device)
            gb_inp = frame[FrameGroup.GB.value].unsqueeze(0).to(device)
            temporal_inp = frame[FrameGroup.TEMPORAL.value].unsqueeze(0).to(device)

            self.model.eval()
            with torch.no_grad():
                wavelet, img = self.model.forward(lr_inp, gb_inp, temporal_inp)

            self.log_image(img[0].detach().cpu(), f'pred_{i}', step)
            self.log_wavelet(wavelet[0].detach().cpu(), f'wavelet_pred_{i}', step)


    def log_gt_images(self):
        """Log the ground truth images to tensorboard.
        """

        for i in self.settings.test_images_idx:
            frame = self.test_dataset[i]

            hr_gt = frame[FrameGroup.HR.value].unsqueeze(0).to(device)
            wavelet = WaveletProcessor.batch_wt(hr_gt)

            self.log_image(hr_gt[0].detach().cpu(), f'gt_{i}', None)
            self.log_wavelet(wavelet[0].detach().cpu(), f'wavelet_gt_{i}', None)
            


    def train(self, num_epochs: int):
        """Train the model

        Args:
            num_epochs (int): Number of epochs to train the model for.
        """
        
        if self.total_epochs == 0:
            self.log_gt_images()

        train_loader = DataLoader(self.train_dataset, batch_size=self.settings.batch_size, shuffle=True)
        validation_loader = DataLoader(self.validation_dataset, batch_size=self.settings.batch_size, shuffle=False)

        for epoch in range(num_epochs):
            # Training

            # Batch loss initialization
            self._batch_loss = None
            self._batch_losses_all = {}

            # Setup threading for training
            train_thread = threading.Thread(target=self._train_batch, args=())
            train_thread.start()
            
            # Get the total number of batches
            num_batches: int = len(train_loader)
            train_epoch_loss: float = 0.0
            train_epoch_losses: Dict[str, float] = {}

            self.model.train()

            # Progress bar
            progress_bar = tqdm(total=num_batches, unit='batch', position=0, leave=True)
            progress_bar.set_description(f'Train Epoch: {self.total_epochs + 1} : {epoch + 1}/{num_epochs}')
            progress_bar.set_postfix_str('Loss: N/A')

            for i, batch in enumerate(train_loader):
                # Wait for the previous batch to finish training and get the loss
                if train_thread.is_alive():
                    train_thread.join()

                # Update the losses and the progress bar
                if self._batch_loss is not None:
                    # Updade the final loss
                    train_epoch_loss += self._batch_loss
                    # Update the total losses
                    for key in self._batch_losses_all:
                        train_epoch_losses[key] = train_epoch_losses.get(key, 0.0) + self._batch_losses_all[key]

                    # Update the progress bar
                    progress_bar.update(1)
                    progress_bar.set_postfix_str(f'Loss: {(train_epoch_loss / i):.4f}')

                # Start the next batch
                train_thread = threading.Thread(target=self._train_batch, args=(batch,))
                train_thread.start()


            # Wait for the last batch to finish training
            if train_thread.is_alive():
                train_thread.join()

            # Update the losses and the progress bar
            # Updade the final loss
            train_epoch_loss += self._batch_loss
            # Update the total losses
            for key in self._batch_losses_all:
                train_epoch_losses[key] = train_epoch_losses.get(key, 0.0) + self._batch_losses_all[key]

            # Update the progress bar
            progress_bar.update(1)
            progress_bar.set_postfix_str(f'Loss: {(train_epoch_loss / num_batches):.4f}')
            # Close the progress bar
            progress_bar.close()

            # Update the losses to be the average
            for key in train_epoch_losses:
                train_epoch_losses[key] /= num_batches
            train_epoch_loss /= num_batches

            # Validation

            # Batch loss initialization
            self._batch_loss = None
            self._batch_losses_all = {}

            # Setup threading for validation
            validation_thread = threading.Thread(target=self._validate_batch, args=())
            validation_thread.start()

            # Losses for validation
            val_epoch_loss: float = 0.0
            val_epoch_losses: Dict[str, float] = {}
            num_batches_val: int = len(validation_loader)

            self.model.eval()

            # Progress bar
            progress_bar = tqdm(total=num_batches_val, unit='batch', position=0, leave=True)
            progress_bar.set_description(f'Valid Epoch: {self.total_epochs + 1} : {epoch + 1}/{num_epochs}')
            progress_bar.set_postfix_str('Loss: N/A')

            for i, batch in enumerate(validation_loader):
                # Wait for the previous batch to finish validation and get the loss
                if validation_thread.is_alive():
                    validation_thread.join()
                    
                # Update the losses and the progress bar
                if self._batch_loss is not None:
                    # Updade the final loss
                    val_epoch_loss += self._batch_loss
                    # Update the total losses
                    for key in self._batch_losses_all:
                        val_epoch_losses[key] = val_epoch_losses.get(key, 0.0) + self._batch_losses_all[key]

                    # Update the progress bar
                    progress_bar.update(1)
                    progress_bar.set_postfix_str(f'Loss: {(val_epoch_loss / i):.4f}')

                # Start the next batch
                validation_thread = threading.Thread(target=self._validate_batch, args=(batch,))
                validation_thread.start()

            # Wait for the last batch to finish validation
            if validation_thread.is_alive():
                validation_thread.join()

            # Update the losses and the progress bar
            # Updade the final loss
            val_epoch_loss += self._batch_loss
            # Update the total losses
            for key in self._batch_losses_all:
                val_epoch_losses[key] = val_epoch_losses.get(key, 0.0) + self._batch_losses_all[key]

            # Update the progress bar
            progress_bar.update(1)
            progress_bar.set_postfix_str(f'Loss: {(val_epoch_loss / num_batches_val):.4f}')
            # Close the progress bar
            progress_bar.close()

            # Update the losses to be the average
            for key in val_epoch_losses:
                val_epoch_losses[key] /= num_batches_val

            val_epoch_loss /= num_batches_val

            # Increment the total epochs
            self.total_epochs += 1

            # Log the losses
            self.log_losses(train_epoch_loss, train_epoch_losses, val_epoch_loss, val_epoch_losses, self.total_epochs)
            
            # Log the test images
            if self.total_epochs % self.settings.output_interval == 0:
                self.log_test_images(self.total_epochs)

            # Save the model checkpoint if the validation loss is the best
            if val_epoch_loss < self.best_validation_loss:
                self.best_validation_loss = val_epoch_loss
                self.save_checkpoint('best.pth')

            # Save the model checkpoint as per frequency in settings
            if self.total_epochs % self.settings.model_save_interval == 0:
                self.save_checkpoint(f'{self.total_epochs}.pth')

            # Step the scheduler
            self.scheduler.step()

        # Save the final model checkpoint
        if self.total_epochs % self.settings.model_save_interval != 0:
            self.save_checkpoint(f'{self.total_epochs}.pth')

    def save_checkpoint(self, file_name: str):
        """Save the model checkpoint.
        """

        ModelUtils.save_checkpoint(self.model, 
                                   self.optimizer, 
                                   self.scheduler,
                                   self.total_epochs, 
                                   float('inf'),
                                   os.path.join(self.settings.model_path(), file_name))


    def load_checkpoint(self, file_name: str):
        """Load the model checkpoint.
        """

        total_epochs, validation_loss = ModelUtils.load_checkpoint(self.model, 
                                                                   self.optimizer, 
                                                                     self.scheduler,
                                                                   os.path.join(self.settings.model_path(), file_name))
        self.total_epochs = total_epochs
        self.best_validation_loss = validation_loss


    def load_best_checkpoint(self):
        """Load the best model checkpoint.
        """    

        self.load_checkpoint('best.pth')

    
    def log_losses(self, train_loss: float, all_train_losses: Dict[str, float], val_loss: float, all_val_losses: Dict[str, float], step: int):
        """Log the losses to tensorboard.
        """

        self.logger.log_scalars('loss', {'train': train_loss, 'val': val_loss}, step)
        for key in all_train_losses:
            self.logger.log_scalars(f'loss_{key}', {'train': all_train_losses[key], 'val': all_val_losses[key]}, step)


    def log_image(self, img: torch.Tensor, tag: str = "", step: int | None = None):
        """Log the image to tensorboard.
        """

        self.logger.log_image(tag, img, step)

    def log_wavelet(self, wavelet: torch.Tensor, tag: str = "", step: int | None = None):
        """Log the wavelet coefficients to tensorboard.
        """
        # The wavelet has 12 channels, 4 coefficients for every RGB channel
        # We need to stack the coefficients in a single image
        # Stack as 
        # Approx, Horizontal
        # Vertical, Diagonal

        # If wavelets have 4 dims, squeeze the first dim
        if wavelet.dim() == 4:
            wavelet = wavelet.squeeze(0)

        _, h, w = wavelet.shape

        # Get the coefficients
        approx = wavelet[0:3, :, :]
        horizontal = wavelet[3:6, :, :]
        vertical = wavelet[6:9, :, :]
        diagonal = wavelet[9:12, :, :]

        # Stack the coefficients
        # Final image will be of shape (batch_size, 3, height * 2, width * 2)
        wavelet_img = torch.zeros((3, h * 2, w * 2), device=wavelet.device)

        wavelet_img[:, :h, :w] = approx
        wavelet_img[:, :h, w:] = vertical
        wavelet_img[:, h:, :w] = diagonal
        wavelet_img[:, h:, w:] = horizontal

        self.logger.log_image(tag, wavelet_img, step)
