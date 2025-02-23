import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import threading
import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image

from .model_utils import ModelUtils
from .losses import CriterionBase , ImageEvaluator , L1Norm
from .models.ModelBase import ModelBase
from .dataset import *
from utils.wdss_logger import NetworkLogger
from utils.wavelet import WaveletProcessor
from config import device, Settings
import json

from enum import Enum
import io

from typing import Dict, Tuple , Any
import json
import cpuinfo

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
        self.l1 = L1Norm()

        # For threading
        self._batch_loss: float | None = None
        self._batch_losses_all: Dict[str, float] = {}

    def _train_batch(self, batch: Dict[str, torch.Tensor] = {}) -> None:
        """Trains the model on a single batch of data. Runs in parallel core while loading the next batch in the main core.
        """
        # If the batch is empty, return
        if not batch:
            return

        lr_inp = batch[FrameGroup.LR.value].to(device)
        gb_inp = batch[FrameGroup.GB.value].to(device)
        temporal_inp = batch[FrameGroup.TEMPORAL.value].to(device)

        hr_gt = batch[FrameGroup.HR.value].to(device)
        hr_wavelet = WaveletProcessor.batch_wt(hr_gt)

        # Zero the gradients
        self.optimizer.zero_grad()

        # Forward pass
        wavelet, img = self.model.forward(lr_inp, gb_inp, temporal_inp, 2.0)

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

        lr_inp = batch[FrameGroup.LR.value].to(device)
        gb_inp = batch[FrameGroup.GB.value].to(device)
        temporal_inp = batch[FrameGroup.TEMPORAL.value].to(device)

        hr_gt = batch[FrameGroup.HR.value].to(device)
        hr_wavelet = WaveletProcessor.batch_wt(hr_gt)

        # Forward pass
        with torch.no_grad():
            wavelet, img = self.model.forward(lr_inp, gb_inp, temporal_inp, 2.0)

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
            frame = self.test_dataset.get_inference_frame(i)

            lr_inp = frame[FrameGroup.LR.value].unsqueeze(0).to(device)
            gb_inp = frame[FrameGroup.GB.value].unsqueeze(0).to(device)
            temporal_inp = frame[FrameGroup.TEMPORAL.value].unsqueeze(0).to(device)

            self.model.eval()
            with torch.no_grad():
                wavelet, img = self.model.forward(lr_inp, gb_inp, temporal_inp, 2.0)

            out, frames = self.test_dataset.preprocessor.postprocess(img, frame[FrameGroup.INFERENCE.value])

            self.log_wavelet(wavelet[0].detach().cpu(), f'Pred_Wavelet_{i}', step)
            for key in frames:
                self.log_image(frames[key][0].detach().cpu(), f'{key}_{i}', step)


    def log_gt_images(self):
        """Log the ground truth images to tensorboard.
        """

        for i in self.settings.test_images_idx:
            frame = self.test_dataset.get_log_frames(i)

            for key in frame:
                if 'Wavelet' in key:
                    self.log_wavelet(frame[key].detach().cpu(), f'{key}_{i}', None)
                else:
                    self.log_image(frame[key].detach().cpu(), f'{key}_{i}', None)
    

    def train(self, num_epochs: int):
        """Train the model

        Args:
            num_epochs (int): Number of epochs to train the model for.
        """
        
        if self.total_epochs == 0:
            self.settings.save_config()
            self.log_gt_images()
            self.log_test_images(0)

        train_loader = DataLoader(self.train_dataset, batch_size=self.settings.batch_size, shuffle=True)
        validation_loader = DataLoader(self.validation_dataset, batch_size=self.settings.batch_size, shuffle=False)

        for epoch in range(num_epochs):
            # Training

            # Batch loss initialization
            self._batch_loss = None
            self._batch_losses_all = {}

            # Setup threading for training
            train_thread = threading.Thread(target=DatasetUtils.wrap_try(self._train_batch), args=())
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
                train_thread = threading.Thread(target=DatasetUtils.wrap_try(self._train_batch), args=(batch,))
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
                        
            self.logger.log_scalar("total_train_loss", train_epoch_loss , self.total_epochs)

            # Update the losses to be the average
            for key in train_epoch_losses:
                train_epoch_losses[key] /= num_batches
            train_epoch_loss /= num_batches

            # Validation

            # Batch loss initialization
            self._batch_loss = None
            self._batch_losses_all = {}

            # Setup threading for validation
            validation_thread = threading.Thread(target=DatasetUtils.wrap_try(self._validate_batch), args=())
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
                validation_thread = threading.Thread(target=DatasetUtils.wrap_try(self._validate_batch), args=(batch,))
                validation_thread.start()

            # Wait for the last batch to finish validation
            if validation_thread.is_alive():
                validation_thread.join()

            # Update the losses and the progress bar
            # Updade the final loss
            val_epoch_loss += self._batch_loss

            # Log the validation total losses
            self.logger.log_scalar("total_val_loss", val_epoch_loss , self.total_epochs)
            
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

            # Log the avg losses
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
            

    def test(self):
        test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)
        
        self.load_best_checkpoint()
        self.model.eval()
        
        all_losses = []  # List to store all losses

        # Wrap the loop with tqdm to show progress
        for i, frame in tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing"):
            lr_inp = frame[FrameGroup.LR.value].to(device)
            gb_inp = frame[FrameGroup.GB.value].to(device)
            temporal_inp = frame[FrameGroup.TEMPORAL.value].to(device)
            hr_gt = frame[FrameGroup.HR.value].to(device)
            hr_wavelet = WaveletProcessor.batch_wt(hr_gt)
            
            l1_wavelet = 0
            psnr_wavelet = 0
            lipps_wavelet = 0
            
            with torch.no_grad():
                wavelet, img = self.model.forward(lr_inp, gb_inp, temporal_inp, 2.0)

            # Calculate the loss using ImageEvaluator's methods
            total_loss, losses = self.criterion.forward(wavelet, hr_wavelet, img, hr_gt)
            
            # Calculate MSE and PSNR using ImageEvaluator methods
            
            l1_loss = self.l1(img, hr_gt)
            psnr = ImageEvaluator.psnr(img, hr_gt)
            lipps = ImageEvaluator.lpips(img, hr_gt)
            
            wavelet_components = [
                (wavelet[:, 0:3, :, :], hr_wavelet[:, 0:3, :, :]),  # Approximation
                (wavelet[:, 3:6, :, :], hr_wavelet[:, 3:6, :, :]),  # Horizontal
                (wavelet[:, 6:9, :, :], hr_wavelet[:, 6:9, :, :]),  # Vertical
                (wavelet[:, 9:12, :, :], hr_wavelet[:, 9:12, :, :])  # Diagonal
            ]
            
            for wavelet_comp, hr_wavelet_comp in wavelet_components:
                psnr_wavelet += ImageEvaluator.psnr(wavelet_comp, hr_wavelet_comp)
                lipps_wavelet += ImageEvaluator.lpips(wavelet_comp, hr_wavelet_comp)
                l1_wavelet += self.l1(wavelet_comp, hr_wavelet_comp)
            
            l1_wavelet /= 4
            psnr_wavelet /= 4
            lipps_wavelet /= 4
            
            losses['l1_image'] = l1_loss.item()
            losses['l1_wavelet'] = l1_wavelet
            
            
            losses['psnr'] = psnr.item()
            losses['psnr_wavelet'] = psnr_wavelet
            
            # Calculate LPIPS using ImageEvaluator method            
            losses['lipps'] = lipps.item()
            losses['lipps_wavelet'] = lipps_wavelet
            
            # Append the losses to the list
            all_losses.append(losses)
        return all_losses  # Return the list of all losses

            

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

    def get_saved_losses(self) -> Tuple[float, Dict[str, float], float, Dict[str, float]]:
        """Get the saved losses.
        """

        losses = os.listdir(self.settings.log_path())
        # Get only folder
        losses = [loss for loss in losses if os.path.isdir(os.path.join(self.settings.log_path(), loss))]
        return losses

    def get_loss_data(self, loss_folder: str) -> Tuple[float, Dict[str, float], float, Dict[str, float]]:
        """Get the loss data from the file.
        """

        loss_file = os.listdir(os.path.join(self.settings.log_path(), loss_folder))[0]
        loss_path = os.path.join(self.settings.log_path(), loss_folder, loss_file)
        
        return self.logger.get_all_scalars(loss_path)
    
    def get_image_data(self) -> Dict[str, list]:
        """Get the image data from the file.
        """
        image_path = os.listdir(self.settings.log_path())
        # Get only folder
        
        image_path = [image for image in image_path if os.path.isfile(os.path.join(self.settings.log_path(), image))]
        
        path = os.path.join(self.settings.log_path(), image_path[0])
        
        return self.logger.get_image_tags(path)
        

    def visualize_losses(self, loss: str, config: Dict[str, Any] = {}):
        """Visualize the loss data with configurable colors, linestyles, clipping, and axis ranges."""
        
        # Fetch loss data
        train_losses = self.logger.get_all_scalars(loss + '_train')
        val_losses = self.logger.get_all_scalars(loss + '_val')

        # Ensure the loss key exists in the dictionaries
        if loss not in train_losses or loss not in val_losses:
            print(f"Loss key '{loss}' not found in data.")
            return

        # Extract step and loss values
        train_steps, train_values = zip(*train_losses[loss]) if train_losses[loss] else ([], [])
        val_steps, val_values = zip(*val_losses[loss]) if val_losses[loss] else ([], [])

        # Convert to NumPy arrays for better handling
        train_values = np.array(train_values)
        val_values = np.array(val_values)

        # Clip values if threshold is specified
        clip_threshold = config.get('clip_threshold', None)
        if clip_threshold:
            train_values = np.clip(train_values, 0, clip_threshold)
            val_values = np.clip(val_values, 0, clip_threshold)

        # Get color and linestyle configurations (fallback to defaults)
        train_color = config.get('train_color', 'blue')
        val_color = config.get('val_color', 'red')
        train_linestyle = config.get('train_linestyle', '-')
        val_linestyle = config.get('val_linestyle', '--')

        # Plot the losses
        plt.figure(figsize=(8, 5))
        plt.plot(train_steps, train_values, label='Train Loss', linestyle=train_linestyle, color=train_color)
        plt.plot(val_steps, val_values, label='Validation Loss', linestyle=val_linestyle, color=val_color)

        # Labels and title
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        if config.get('title', True):
            plt.title(f'Loss Curve for {loss}')
            
        plt.legend()

        # Grid for better visualization
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Rotate x-axis labels for better visibility
        plt.xticks(rotation=45)

        # Set axis limits if provided in config
        if 'xlim' in config:
            plt.xlim(config['xlim'])
        if 'ylim' in config:
            plt.ylim(config['ylim'])

        # Show the plot
        plt.show()



        
    def visualize_image(self, tag: str , step : int) :
        """Visualize the image data."""
        
        # Fetch image data
        images = self.get_image_data()
        
        # Ensure the tag exists in the list
        if tag not in images:
            print(f"Tag '{tag}' not found in data.")
            return
        
        # Get the images for the tag
        image_data = self.logger.get_images_by_tag(self.settings.log_path(), tag)
        
        # Filter images by step
        image_data = [img for img in image_data if img[0] == step]
        
        # Ensure images are found
        if not image_data:
            print(f"No images found for tag '{tag}' at step {step}.")
            return
        
        # Display the images
        for i, (step, img) in enumerate(image_data):
            plt.figure(figsize=(8, 5))
            plt.imshow(plt.imread(io.BytesIO(img)))
            plt.title(f'{tag} at step {step}')
            plt.axis('off')
            plt.show()

        

    def test_images(self, index: int, save: bool = False):
        """Test the images and display results."""
        
        self.load_best_checkpoint()
        self.model.eval()

        # Get test image
        frame = self.test_dataset[index]
        lr_inp = frame[FrameGroup.LR.value].unsqueeze(0).to(device)
        gb_inp = frame[FrameGroup.GB.value].unsqueeze(0).to(device)
        temporal_inp = frame[FrameGroup.TEMPORAL.value].unsqueeze(0).to(device)
        hr_gt = frame[FrameGroup.HR.value].unsqueeze(0).to(device)

        # Compute wavelet transform on ground truth
        hr_wavelet = WaveletProcessor.batch_wt(hr_gt)

        with torch.no_grad():
            wavelet, img = self.model.forward(lr_inp, gb_inp, temporal_inp, 2.0)

        # Convert tensors to CPU for visualization
        lr_img = lr_inp[0].detach().cpu()
        hr_img = hr_gt[0].detach().cpu()
        pred_img = img[0].detach().cpu()

        # Display original images
        ImageUtils.display_images([lr_img, hr_img, pred_img], ['LR', 'HR', 'Pred'])

        # Create save directory for this index
        save_dir = f"saved_tests/{index}"
        os.makedirs(save_dir, exist_ok=True)

        # Save images if enabled
        if save:
            save_image(lr_img, f"{save_dir}/lr.png")
            save_image(hr_img, f"{save_dir}/hr.png")
            save_image(pred_img, f"{save_dir}/pred.png")

        # Display and save wavelet components
        for i, label in enumerate(["Approximation", "Horizontal", "Vertical", "Diagonal"]):
            hr_wavelet_img = hr_wavelet[0, i * 3:(i + 1) * 3, :, :].detach().cpu()
            pred_wavelet_img = wavelet[0, i * 3:(i + 1) * 3, :, :].detach().cpu()

            # Show images without modifications
            ImageUtils.display_images([hr_wavelet_img, pred_wavelet_img], [f'HR {label}', f'Pred {label}'])

            if save:
                # Normalize wavelets for saving (min-max scaling to [0,1])
                def normalize_image(img):
                    min_val, max_val = img.min(), img.max()
                    return (img - min_val) / (max_val - min_val + 1e-8)  # Avoid division by zero

                hr_wavelet_img = normalize_image(hr_wavelet_img)
                pred_wavelet_img = normalize_image(pred_wavelet_img)

                # Save images
                save_image(hr_wavelet_img, f"{save_dir}/hr_wavelet_{label.lower()}.png")
                save_image(pred_wavelet_img, f"{save_dir}/pred_wavelet_{label.lower()}.png")

    def profile_performance(self):

        frame = self.test_dataset[0]
        lr_frame = frame[FrameGroup.LR.value].unsqueeze(0).to(device)
        gb_buffer = frame[FrameGroup.GB.value].unsqueeze(0).to(device)
        temporal_frame = frame[FrameGroup.TEMPORAL.value].unsqueeze(0).to(device)

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.settings.log_path()),
            record_shapes=True,
            with_stack=True,
            profile_memory=True
        ) as profiler:
            for _ in range(5):  # Run multiple steps to collect profiling data)
                with torch.no_grad():
                    self.model.forward(lr_frame, gb_buffer, temporal_frame, 2.0)
                profiler.step()

        # Collect profiling results
        profiler_results = profiler.key_averages().table(sort_by="cuda_time_total")


        # Save profiling results to a file
        profiler_results_path = os.path.join(self.settings.log_path(), 'profiler_results.txt')
        with open(profiler_results_path, 'w') as f:
            f.write("Profiler Results:\n")
            f.write(profiler_results)

            # Save device details
            device_details = {
                "device_id": str(device),
                "device_name": torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU",
                "total_memory": torch.cuda.get_device_properties(device).total_memory if torch.cuda.is_available() else "N/A",
                "memory_allocated": torch.cuda.memory_allocated(device) if torch.cuda.is_available() else "N/A",
                "memory_reserved": torch.cuda.memory_reserved(device) if torch.cuda.is_available() else "N/A",
                "cpu_name": cpuinfo.get_cpu_info()['brand_raw'],
                "cpu_architecture": cpuinfo.get_cpu_info()['arch'],
                "cpu_cores": cpuinfo.get_cpu_info()['count'],
                "cpu_threads": os.cpu_count()
            }

            f.write("\n\nDevice Details:\n")
            json.dump(device_details, f, indent=4)
