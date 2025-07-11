import torch
from tqdm import tqdm
from multiprocessing.pool import ThreadPool, AsyncResult
import matplotlib.pyplot as plt
import os

from .model_utils import ModelUtils
from .losses import CriterionBase
from .models.ModelBase import ModelBase
from .dataset import *
from utils.brdf import BRDFProcessor
from utils.wdss_logger import NetworkLogger
from utils.wavelet import WaveletProcessor
from config import device, Settings
import json
from utils.image_utils import ImageUtils
from .image_evaluator import exp_norm, reinhard_norm, ImageEvaluator

from typing import Dict, Tuple, Any, Optional

class Trainer:
    def __init__(
        self,
        settings: Settings,
        model: ModelBase,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        criterion: CriterionBase,
        train_dataset: WDSSDataset,
        val_dataset: WDSSDataset,
        test_dataset: WDSSDataset,
    ):
        super().__init__()

        self.settings = settings
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self._thread_pool = ThreadPool(processes=1)
        self.total_epochs = 0
        self.logger = NetworkLogger(settings.log_path())
        self.best_val_loss = float('inf')

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.settings['batch_size'],
            shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.settings['batch_size'],
            shuffle=False
        )

    def train(self, epochs: int = 1, no_log_gt: bool = False) -> None:
        """Train the model for a specified number of epochs.
        """
        
        self.settings.save_config()

        if self.total_epochs == 0 and not no_log_gt:
            self.log_gt_images()
            self.log_test_frames()

        for epoch in range(epochs):
            train_loss, train_metrics = self._train_single_epoch(epoch, epochs)
            val_loss, val_metrics = self._validate_single_epoch(epoch, epochs)

            self.scheduler.step()

            # Increment the total epochs
            self.total_epochs += 1

            # Log the losses
            self.log_losses(
                train_loss,
                val_loss,
                train_metrics,
                val_metrics,
                step=self.total_epochs
            )

            # Save the model checkpoint
            self.save_checkpoint('latest.pth', val_loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best.pth', val_loss)

            if self.total_epochs % self.settings.model_save_interval == 0:
                self.save_checkpoint(f'epoch_{self.total_epochs}.pth', val_loss)

            if self.total_epochs % self.settings.image_log_interval == 0:
                self.log_test_frames()

    def _train_batch(self, batch: Dict[str, torch.Tensor | Dict[str, torch.Tensor]] = {}) -> Tuple[Optional[float], Dict[str, float]]:
        """Train the model on a batch of data and return the loss and metrics.
        """

        if not batch:
            return None, {}
        
        batch = WDSSDataset.batch_to_device(batch, device)
        lr_inp = batch[FrameGroup.LR_INP.value]
        gb_inp = batch[FrameGroup.GB_INP.value]
        temporal_inp = batch[FrameGroup.TEMPORAL_INP.value]

        hr_gt = batch[FrameGroup.GT.value]
        gt_wavelet = WaveletProcessor.batch_wt(hr_gt)

        # Zero the gradients
        self.optimizer.zero_grad()

        upscale_factor: float = hr_gt.shape[-2] / lr_inp.shape[-2]

        # Forward pass
        wavelet, img = self.model.forward(lr_inp, gb_inp, temporal_inp, upscale_factor)

        # Compute the loss
        total_loss, metrics = self.criterion.forward(
            img,
            hr_gt,
            wavelet,
            gt_wavelet,
            batch
        )

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)

        # Update the weights
        self.optimizer.step()

        metrics_float: Dict[str, float] = {k: v.item() for k, v in metrics.items()}
        return total_loss.item(), metrics_float
    

    def _validate_batch(self, batch: Dict[str, torch.Tensor | Dict[str, torch.Tensor]] = {}) -> Tuple[Optional[float], Dict[str, float]]:
        """Validate the model on a batch of data and return the loss and metrics.
        """
        if not batch:
            return None, {}

        batch = WDSSDataset.batch_to_device(batch, device)
        lr_inp = batch[FrameGroup.LR_INP.value]
        gb_inp = batch[FrameGroup.GB_INP.value]
        temporal_inp = batch[FrameGroup.TEMPORAL_INP.value]

        hr_gt = batch[FrameGroup.GT.value]
        gt_wavelet = WaveletProcessor.batch_wt(hr_gt)

        upscale_factor: float = hr_gt.shape[2] / lr_inp.shape[2]

        with torch.no_grad():
            # Forward pass
            wavelet, img = self.model.forward(lr_inp, gb_inp, temporal_inp, upscale_factor)

            # Compute the loss
            total_loss, metrics = self.criterion.forward(
                img,
                hr_gt,
                wavelet,
                gt_wavelet,
                batch
            )

        metrics_float: Dict[str, float] = {k: v.item() for k, v in metrics.items()}
        return total_loss.item(), metrics_float
    
    def _train_single_epoch(self, curr_epoch: int = 0, epochs: int = 0) -> Tuple[float, Dict[str, float]]:
        """Train the model for a single epoch and return the average loss and metrics.
        """
        # Initialize variables
        total_loss: float = 0.0
        total_metrics: Dict[str, float] = {}
        num_batches = len(self.train_loader)

        # Initialize the progress bar
        progress_bar = tqdm(
            total=num_batches,
            unit='batch',
            position=0,
            leave=True
        )
        progress_bar.set_description(f'Train Epoch: {self.total_epochs + 1} : {curr_epoch + 1}/{epochs}')
        progress_bar.set_postfix(loss=float('nan'))

        # Set the model to training mode
        self.model.train()

        async_result: AsyncResult = self._thread_pool.apply_async(
            self._train_batch,
            args=({}) # Pass an empty dictionary to avoid errors
        )

        for i, batch in enumerate(self.train_loader):
            # Wait for the previous batch to finish
            loss, metrics = async_result.get()
            if loss is not None:
                total_loss += loss
                for key, value in metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = 0.0
                    total_metrics[key] += value

                # Update the progress bar
                progress_bar.update(1)
                # progress_bar.set_postfix(loss=total_loss / (i), **{
                #     k: v / (i) for k, v in total_metrics.items()
                # })
                progress_bar.set_postfix(loss=total_loss / (i))

            # Start the next batch
            async_result = self._thread_pool.apply_async(
                func=self._train_batch,
                args=(batch,)
            )

            self.train_dataset.set_random_upscale_factor()

        # Wait for the last batch to finish
        loss, metrics = async_result.get()
        total_loss += loss
        for key, value in metrics.items():
            if key not in total_metrics:
                total_metrics[key] = 0.0
            total_metrics[key] += value

        # Average the metrics
        total_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        total_loss /= num_batches

        # Update the learning progress bar
        progress_bar.update(1)
        # progress_bar.set_postfix(loss=total_loss, **total_metrics)
        progress_bar.set_postfix(loss=total_loss)
        # Close the progress bar
        progress_bar.close()

        return total_loss, total_metrics
    
    def _validate_single_epoch(self, curr_epoch: int = 0, epochs: int = 0) -> Tuple[float, Dict[str, float]]:
        """Validate the model for a single epoch and return the average loss and metrics.
        """
        # Initialize variables
        total_loss: float = 0.0
        total_metrics: Dict[str, float] = {}
        num_batches = len(self.val_loader)

        # Initialize the progress bar
        progress_bar = tqdm(
            total=num_batches,
            unit='batch',
            position=0,
            leave=True
        )
        progress_bar.set_description(f'Validation Epoch: {self.total_epochs + 1} : {curr_epoch + 1}/{epochs}')
        progress_bar.set_postfix(loss=float('nan'), **total_metrics)

        # Set the model to evaluation mode
        self.model.eval()

        async_result: AsyncResult = self._thread_pool.apply_async(
            self._validate_batch,
            args=({}) # Pass an empty dictionary to avoid errors
        )

        for i, batch in enumerate(self.val_loader):
            # Wait for the previous batch to finish
            loss, metrics = async_result.get()
            if loss is not None:
                total_loss += loss
                for key, value in metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = 0.0
                    total_metrics[key] += value

                # Update the progress bar
                progress_bar.update(1)
                # progress_bar.set_postfix(loss=total_loss / (i), **{
                #     k: v / (i) for k, v in total_metrics.items()
                # })
                progress_bar.set_postfix(loss=total_loss / (i))

                torch.cuda.empty_cache()

            # Start the next batch
            async_result = self._thread_pool.apply_async(
                func=self._validate_batch,
                args=(batch,)
            )

            self.val_dataset.set_random_upscale_factor()

        # Wait for the last batch to finish
        loss, metrics = async_result.get()
        total_loss += loss if loss is not None else 0.0
        for key, value in metrics.items():
            if key not in total_metrics:
                total_metrics[key] = 0.0
            total_metrics[key] += value

        # Average the metrics
        total_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        total_loss /= num_batches

        # Update the learning progress bar
        progress_bar.update(1)
        # progress_bar.set_postfix(loss=total_loss, **total_metrics)
        progress_bar.set_postfix(loss=total_loss)
        
        # Close the progress bar
        progress_bar.close()

        return total_loss, total_metrics
    
    def save_checkpoint(self, file_name: str, val_loss: float = float('inf')) -> None:
        """Save the model checkpoint to a file.
        """
        ModelUtils.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.total_epochs,
            validation_loss=val_loss,
            checkpoint_path=os.path.join(self.settings.model_path(), file_name)
        )

    def load_checkpoint(self, file_name: str = None, epoch: int = None) -> None:
        """Load the model checkpoint from a file.
        """
        if not file_name and not epoch:
            raise ValueError("Either file_name or epoch must be provided to load a checkpoint.")
        if not file_name:
            file_name = f'epoch_{epoch}.pth'

        total_epochs, validation_loss = ModelUtils.load_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            checkpoint_path=os.path.join(self.settings.model_path(), file_name)
        )
        self.total_epochs = total_epochs
        self.best_val_loss = validation_loss

    def load_best_checkpoint(self) -> None:
        """Load the best model checkpoint from a file.
        """
        self.load_checkpoint('best.pth')

    def load_latest_checkpoint(self) -> None:
        """Load the latest model checkpoint from a file.
        """
        self.load_checkpoint('latest.pth')

    def log_gt_images(self):
        """Log the GT and LR images to the TensorBoard.
        """

        for index, upscale_factor in self.settings.test_images_idx:
            frame = self.test_dataset.get_log_frame(index, upscale_factor, True)

            for key in frame:
                img = frame[key]
                if 'Wavelet' in key:
                    img = ImageUtils.stack_wavelet(img)

                if 'HR' in key:
                    self.log_image(
                        img.detach().cpu(),
                        f'{key}/{index}',
                        None
                    )
                else:
                    self.log_image(
                        img.detach().cpu(),
                        f'{key}/{index}/{upscale_factor:.1f}x',
                        None
                    )

    # @wrap_try
    def log_test_frames(self) -> None:
        """Log the test frames to the TensorBoard.
        """

        for index, upscale_factor in self.settings.test_images_idx:
            frame = self._inference_for_logging(index, upscale_factor)

            for key in frame:
                img = frame[key]
                if 'Wavelet' in key:
                    img = ImageUtils.stack_wavelet(img)

                self.log_image(
                    img.detach().cpu(),
                    f'{key}/{index}/{upscale_factor:.1f}x',
                    self.total_epochs
                )

    def inference(self, index: int, upscale_factor: float) -> Dict[str, torch.Tensor]:
        """Run inference on a single image and return the output.
        """    
        res: Dict[str, torch.Tensor] = {}
        frame = self.test_dataset.get_item(index, upscale_factor, True)
        frame = WDSSDataset.batch_to_device(frame, device)
        frame = WDSSDataset.unsqueeze_batch(frame)

        lr_inp = frame[FrameGroup.LR_INP.value]
        gb_inp = frame[FrameGroup.GB_INP.value]
        temporal_inp = frame[FrameGroup.TEMPORAL_INP.value]

        upscale_factor: float = frame[FrameGroup.GT.value].shape[-2] / lr_inp.shape[-2] 

        self.model.eval()
        with torch.no_grad():
            wavelet, img = self.model.forward(lr_inp, gb_inp, temporal_inp, upscale_factor)

        out = self.test_dataset.preprocessor.postprocess(
            img,
            extra=frame[FrameGroup.EXTRA.value]
        )

        res['Pred'] = img
        res['PredWavelet'] = wavelet
        res['PredTonemapped'] = self.test_dataset.preprocessor.tonemap(out)
        res['PredReconstructed'] = out

        return res
    
    def _inference_for_logging(self, index: int, upscale_factor: float) -> Dict[str, torch.Tensor]:
        infered = self.inference(index, upscale_factor)
        infered['Pred'] = exp_norm(infered['Pred'])
        infered['PredWavelet'] = exp_norm(infered['PredWavelet'].clamp(0.0, None))
        infered['PredReconstructed'] = exp_norm(infered['PredReconstructed'])
        return infered

    
    def evaluate_single(self, index: int, upscale_factor: float = 2.0, dataset: WDSSDataset | None = None, evaluate_model: bool = True, evaluate_bilinear: bool = False, evaluate_bilinear_demod: bool = False, no_patch: bool = True) -> Tuple[Dict[str, float] | None, Dict[str, float] | None, Dict[str, float] | None]:
        """Compute the metrics for a single image.

        Args:
            index (int): Index of the image to evaluate.
            upscale_factor (float): Upscale factor.
            dataset (WDSSDataset | None): Dataset to use for evaluation. If None, use the test dataset.
            evaluate_model (bool): Whether to evaluate the model.
            evaluate_bilinear (bool): Whether to evaluate the bilinear interpolation.
            evaluate_bilinear_demod (bool): Whether to evaluate the bilinear demodulation.
            no_patch (bool): Whether to use the full image or a patch.


        Returns:
            Evaluation results for the model, bilinear interpolation, and bilinear interpolation of demodulated image.
        """
        import time

        if dataset is None:
            dataset = self.test_dataset

        model_result: None | Dict[str, float] = None
        bilinear_result: None | Dict[str, float] = None
        bilinear_demod_result: None | Dict[str, float] = None

        frame = dataset.get_item(index, upscale_factor, no_patch)
        frame = WDSSDataset.batch_to_device(frame, device)
        frame = WDSSDataset.unsqueeze_batch(frame)

        lr_inp = frame[FrameGroup.LR_INP.value]
        gb_inp = frame[FrameGroup.GB_INP.value]
        temporal_inp = frame[FrameGroup.TEMPORAL_INP.value]
        hr_gt = frame[FrameGroup.GT.value]

        upscale_factor = hr_gt.shape[-2] / lr_inp.shape[-2]
        
        gt_remod = dataset.preprocessor.postprocess(
            hr_gt,
            extra=frame[FrameGroup.EXTRA.value]
        )
        gt_tonemapped = dataset.preprocessor.tonemap(gt_remod)

        if evaluate_model:
            self.model.eval()
            with torch.no_grad():
                _, img = self.model.forward(lr_inp, gb_inp, temporal_inp, upscale_factor)

            remod = dataset.preprocessor.postprocess(
                img,
                extra=frame[FrameGroup.EXTRA.value]
            )
            tonemapped = dataset.preprocessor.tonemap(remod)
            model_result = ImageEvaluator.evaluate(tonemapped, gt_tonemapped)

        if evaluate_bilinear:
            try:
                bilinear_inp = BRDFProcessor.brdf_remodulate(lr_inp, frame[FrameGroup.EXTRA.value]['BRDF_LR'])
            except:
                bilinear_inp = lr_inp
            bilinear = ImageUtils.upsample(bilinear_inp, upscale_factor)
            bilinear = dataset.preprocessor.tonemap(bilinear)

            bilinear_result = ImageEvaluator.evaluate(bilinear, gt_tonemapped)

        if evaluate_bilinear_demod:
            bilinear = ImageUtils.upsample(lr_inp, upscale_factor)
            bilinear = dataset.preprocessor.postprocess(bilinear, extra=frame[FrameGroup.EXTRA.value])
            bilinear = dataset.preprocessor.tonemap(bilinear)

            bilinear_demod_result = ImageEvaluator.evaluate(bilinear, gt_tonemapped)

        return model_result, bilinear_result, bilinear_demod_result
    
    def evaluate(self, range: Tuple[int, int] | None = None, upscale_factors: List[float] | None = None, dataset: WDSSDataset | None = None, evaluate_model: bool = True, evaluate_bilinear: bool = False, evaluate_bilinear_demod: bool = False, 
        print_results: bool = True, log_file: str = "", divide_zips: bool = True, store_individual: bool = False
    ) -> Tuple[Dict[float, Tuple[Dict[str, float] | None, Dict[str, float] | None, Dict[str, float] | None]], Dict[float, List[Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]]] | None]:
        ...

    def log_losses(self, train_loss: float, val_loss: float, train_metrics: Dict[str, float], val_metrics: Dict[str, float], step: int | None = None) -> None:
        """Log the training and validation losses and metrics to the TensorBoard.
        """
        self.logger.log_scalars('loss', {'train': train_loss, 'val': val_loss}, step)
        for key in train_metrics:
            self.logger.log_scalars(key, {'train': train_metrics[key], 'val': val_metrics.get(key, float('nan'))}, step)

    def log_image(self, img: torch.Tensor, tag: str = "", step: int | None = None):
        """Log an image to the TensorBoard.
        """
        if img.dim() == 4:
            img = img.squeeze(0)
        self.logger.log_image(tag, img, step)