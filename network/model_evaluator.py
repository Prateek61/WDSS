import torch
from .models.ModelBase import ModelBase
from utils.image_utils import ImageUtils
from .image_evaluator import ImageEvaluator
from utils.preprocessor import Preprocessor
from .dataset import *
from config import device

import json
from tqdm import tqdm
import sys
import os

from typing import Dict

class EvaluationResults:
    """Class to store evaluation results of the data.
    """
    def __init__(
        self,
        start: int = 0,
        end: int = 0,
        name: str = "",
        average: Dict[float, Dict[str, float]] | None = None, # Average metrics for every upscale factor
        per_frame_metrics: Dict[float, List[Dict[str, float]]] | None = None, # Metrics for every frame
        per_zip: Dict[float, Dict[str, Dict[str, float]]] | None = None, # Average metrics for every zip file
    ):
        self.name = name
        self.start = start
        self.end = end
        self.average = average if average is not None else {}
        self.per_frame_metrics = per_frame_metrics if per_frame_metrics is not None else {}
        self.per_zip = per_zip if per_zip is not None else {}

    @property
    def total_frames(self) -> int:
        """Get the total number of frames evaluated.
        """
        return self.end - self.start
    
    @property
    def total_zip_files(self) -> int:
        """Get the total number of zip files evaluated.        
        """
        if self.per_zip is None:
            return 0
        for _, value in self.per_zip.items():
            return len(value)
        
    def frame(self, index: int, upscale_factor: float | None = None) -> Dict[str, float] | Dict[float, Dict[str, float]]:
        """Get the metrics for a given frame.
        """
        if upscale_factor is not None:
            return self.per_frame_metrics[upscale_factor][index - self.start]
        return {
            k: v[index - self.start] for k, v in self.per_frame_metrics.items()
        }
    
    def metrics_names(self) -> List[str]:
        """Get the names of the metrics.
        """
        return list(self.average.values())[0].keys()
    
    def __str__(self) -> str:
        """Get the string representation of the evaluation results.
        """
        res = f"Evaluation Results ({self.name})\n"
        # Print a table of the average metrics for each upscale factor
        res += f"{'Upscale Factor':<15}"
        for metric in self.metrics_names():
            res += f"{metric:<15}"
        res += "\n"
        for upscale_factor, metrics in self.average.items():
            res += f"{upscale_factor:<15}"
            for metric in self.metrics_names():
                res += f"{metrics[metric]:<15.4f}"
            res += "\n"
        res += "\n"

        return res
    
    def save_json(self, path: str) -> None:
        """Save the evaluation results to a JSON file.
        """
        data = {
            'name': self.name,
            'start': self.start,
            'end': self.end,
            'average': self.average,
            'per_zip': self.per_zip,
            'per_frame_metrics': self.per_frame_metrics
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        
    def load_json(self, path: str) -> None:
        """Load the evaluation results from a JSON file.
        """
        with open(path, 'r') as f:
            data = json.load(f)
            self.name = data['name']
            self.start = data['start']
            self.end = data['end']
            self.average = data['average']
            self.per_zip = data['per_zip']
            self.per_frame_metrics = data['per_frame_metrics']


class ModelEvaluator:
    @staticmethod
    def evaluate_model_frame(model: ModelBase, frame: Dict[str, torch.Tensor | Dict[str, torch.Tensor]], preprocessor: Preprocessor
    ) -> Dict[str, float]:
        """Evaluate the model on a given frame.
        """

        # Get the input and target images
        lr_inp = frame[FrameGroup.LR_INP.value] 
        gb_inp = frame[FrameGroup.GB_INP.value]
        temporal_inp = frame[FrameGroup.TEMPORAL_INP.value]
        gt = frame[FrameGroup.GT.value]
        extra = frame[FrameGroup.EXTRA.value]

        gt_remod = preprocessor.postprocess(gt, extra=extra)
        gt_tonemapped = preprocessor.tonemap(gt_remod)
        
        upscale_factor = gt.shape[-2] / lr_inp.shape[-2]

        # Forward pass through the model
        with torch.no_grad():
            _, out = model.forward(lr_inp, gb_inp, temporal_inp, upscale_factor)

            # Post-process the output
            out_remod = preprocessor.postprocess(out, extra=extra)
            out_tonemapped = preprocessor.tonemap(out_remod)

            # Calculate evaluation metrics
            return ImageEvaluator.evaluate(input=out_tonemapped, target=gt_tonemapped)

    @staticmethod
    def evaluate_bilinear_frame(frame: Dict[str, torch.Tensor | Dict[str, torch.Tensor]], preprocessor: Preprocessor) -> Dict[str, float]:
        """Evaluate the bilinear interpolation on a given frame.
        """
        # Get the input and target images
        lr_inp = frame[FrameGroup.LR_INP.value]
        gt = frame[FrameGroup.GT.value]
        extra = frame[FrameGroup.EXTRA.value]

        upscale_factor: float = gt.shape[-2] / lr_inp.shape[-2]

        gt_remod = preprocessor.postprocess(gt, extra=extra)
        gt_tonemapped = preprocessor.tonemap(gt_remod)

        lr_remod = preprocessor.postprocess_lr(lr_inp, extra=extra)
        lr_tonemapped = preprocessor.tonemap(lr_remod)
        # Upscale the low-resolution image using bilinear interpolation
        lr_upscaled = ImageUtils.upsample(lr_tonemapped, upscale_factor)

        # Calculate evaluation metrics
        return ImageEvaluator.evaluate(input=lr_upscaled, target=gt_tonemapped)
    
    @staticmethod
    def evaluate_bilinear_demod_frame(frame: Dict[str, torch.Tensor | Dict[str, torch.Tensor]], preprocessor: Preprocessor) -> Dict[str, float]:
        """Evaluate the bilinear interpolation on a given frame.
        """
        # Get the input and target images
        lr_inp = frame[FrameGroup.LR_INP.value]
        gt = frame[FrameGroup.GT.value]

        upscale_factor: float = gt.shape[-2] / lr_inp.shape[-2]

        gt_remod = preprocessor.postprocess(gt, extra=frame[FrameGroup.EXTRA.value])
        gt_tonemapped = preprocessor.tonemap(gt_remod)

        upsampled = ImageUtils.upsample(lr_inp, upscale_factor)
        ups_remod = preprocessor.postprocess(upsampled, extra=frame[FrameGroup.EXTRA.value])
        ups_tonemapped = preprocessor.tonemap(ups_remod)

        # Calculate evaluation metrics
        return ImageEvaluator.evaluate(input=ups_tonemapped, target=gt_tonemapped)
    
    @staticmethod
    def evaluate(
        model: ModelBase,
        dataset: WDSSDataset,
        upscale_factors: List[float] | None = None,
        frame_range: Tuple[int, int] | None = None 
    ) -> EvaluationResults:
        """Evaluate the model on the given dataset.
        """

        if frame_range is None:
            start = 0
            end = dataset.total_frames
            total_frames = dataset.total_frames
        else:
            start, end = frame_range
            total_frames = end - start

        if upscale_factors is None:
            upscale_factors = dataset.upscale_factors
        upscale_factors = sorted(upscale_factors)

        # Initialize the evaluation results
        results = EvaluationResults(start=start, end=end, name="Model Evaluation")

        # Initialize lpips
        ImageEvaluator.initialize_lpips()

        for upscale_factor in upscale_factors:
            average_metrics = {k : 0.0 for k in ImageEvaluator.evaluation_metrics}
            curr_zip_metrics = {k: 0.0 for k in ImageEvaluator.evaluation_metrics}
            zip_metrics = {}
            per_frame_metrics = []
            zip_start = start

            progress_bar = tqdm(
                total=total_frames,
                unit="frame",
                position=0,
                leave=True,
                desc=f"Evaluating {upscale_factor}x"
            )
            progress_bar.set_postfix(**average_metrics)

            for i in range(start, end):
                # Get the frame from the dataset
                frame = dataset.get_item(i, upscale_factor=upscale_factor, no_patch=True)
                frame = dataset.batch_to_device(frame)
                frame = dataset.unsqueeze_batch(frame)

                # Get the evaluation metrics for the frame
                curr_metrics = ModelEvaluator.evaluate_model_frame(model=model, frame=frame, preprocessor=dataset.preprocessor)
                # Add the metrics to the average
                for k, v in curr_metrics.items():
                    average_metrics[k] += v
                    curr_zip_metrics[k] += v
                # Add the metrics to the per frame results
                per_frame_metrics.append(curr_metrics)

                # Update the progress bar
                progress_bar.update(1)
                progress_bar.set_postfix(**{
                    k: v / (i + 1 - start) for k, v in average_metrics.items()
                })

                # If the current frame is the last frame of the zip file, save the metrics
                if (i + 1) % dataset.frames_per_zip == 0 or i == end - 1:
                    curr_zip_metrics = {k: v / (i - zip_start + 1) for k, v in curr_zip_metrics.items()}
                    zip_metrics[dataset.compressed_files[i // dataset.frames_per_zip]] = curr_zip_metrics
                    zip_start = i + 1
                    # Reset the current zip metrics
                    curr_zip_metrics = {k: 0.0 for k in ImageEvaluator.evaluation_metrics}

            # Close the progress bar
            progress_bar.close()
            # Calculate the average metrics for the upscale factor
            average_metrics = {k: v / total_frames for k, v in average_metrics.items()}
            # Add the metrics to the results
            results.average[upscale_factor] = average_metrics
            results.per_frame_metrics[upscale_factor] = per_frame_metrics
            results.per_zip[upscale_factor] = zip_metrics

        return results
    
    @staticmethod
    def evaluate_bilinear(
        dataset: WDSSDataset,
        upscale_factors: List[float] | None = None,
        frame_range: Tuple[int, int] | None = None 
    ) -> Tuple[EvaluationResults, EvaluationResults]:
        """Evaluate the bilinear interpolation on the given dataset.

        Returns:
            Tuple[EvaluationResults, EvaluationResults]: The evaluation results for the bilinear interpolation and the bilinear interpolation with of demodulated frame.
        """

        if frame_range is None:
            start = 0
            end = dataset.total_frames
            total_frames = dataset.total_frames
        else:
            start, end = frame_range
            total_frames = end - start

        if upscale_factors is None:
            upscale_factors = dataset.upscale_factors
        upscale_factors = sorted(upscale_factors)

        # Initialize the evaluation results
        results_bilinear = EvaluationResults(start=start, end=end, name="Bilinear Evaluation")
        results_bilinear_demod = EvaluationResults(start=start, end=end, name="Bilinear Demod Evaluation")

        # Initialize lpips
        ImageEvaluator.initialize_lpips()

        for upscale_factor in upscale_factors:
            average_metrics_bilinear = {k : 0.0 for k in ImageEvaluator.evaluation_metrics}
            average_metrics_bilinear_demod = {k : 0.0 for k in ImageEvaluator.evaluation_metrics}
            curr_zip_metrics_bilinear = {k: 0.0 for k in ImageEvaluator.evaluation_metrics}
            curr_zip_metrics_bilinear_demod = {k: 0.0 for k in ImageEvaluator.evaluation_metrics}
            zip_metrics_bilinear = {}
            zip_metrics_bilinear_demod = {}
            per_frame_metrics_bilinear = []
            per_frame_metrics_bilinear_demod = []
            zip_start = start

            progress_bar = tqdm(
                total=total_frames,
                unit="frame",
                position=0,
                leave=True,
                desc=f"Evaluating bilinear {upscale_factor}x"
            )
            progress_bar.set_postfix(loss=float('nan'))

            for i in range(start, end):
                # Get the frame from the dataset
                frame = dataset.get_item(i, upscale_factor=upscale_factor, no_patch=True)
                frame = dataset.batch_to_device(frame)
                frame = dataset.unsqueeze_batch(frame)

                # Get the evaluation metrics for the bilinear interpolation
                curr_metrics_bilinear = ModelEvaluator.evaluate_bilinear_frame(frame, dataset.preprocessor)
                curr_metrics_bilinear_demod = ModelEvaluator.evaluate_bilinear_demod_frame(frame, dataset.preprocessor)

                for k in curr_metrics_bilinear.keys():
                    average_metrics_bilinear[k] += curr_metrics_bilinear[k]
                    curr_zip_metrics_bilinear[k] += curr_metrics_bilinear[k]
                    average_metrics_bilinear_demod[k] += curr_metrics_bilinear_demod[k]
                    curr_zip_metrics_bilinear_demod[k] += curr_metrics_bilinear_demod[k]
                # Add the metrics to the per frame results
                per_frame_metrics_bilinear.append(curr_metrics_bilinear)
                per_frame_metrics_bilinear_demod.append(curr_metrics_bilinear_demod)

                # Update the progress bar
                progress_bar.update(1)
                progress_bar.set_postfix(**{
                    f"bilinear_{k}": v / (i + 1 - start) for k, v in average_metrics_bilinear.items()
                })

                # If the current frame is the last frame of the zip file, save the metrics
                if (i + 1) % dataset.frames_per_zip == 0 or i == end - 1:
                    zip_metrics_bilinear[dataset.compressed_files[i // dataset.frames_per_zip]] = curr_zip_metrics_bilinear.copy()
                    zip_metrics_bilinear_demod[dataset.compressed_files[i // dataset.frames_per_zip]] = curr_zip_metrics_bilinear_demod.copy()
                    curr_zip_metrics_bilinear_demod = {k: v / (i - zip_start + 1) for k, v in curr_zip_metrics_bilinear_demod.items()}
                    zip_metrics_bilinear[dataset.compressed_files[i // dataset.frames_per_zip]] = curr_zip_metrics_bilinear
                    zip_metrics_bilinear_demod[dataset.compressed_files[i // dataset.frames_per_zip]] = curr_zip_metrics_bilinear_demod
                    zip_start = i + 1
                    # Reset the current zip metrics
                    curr_zip_metrics_bilinear = {k: 0.0 for k in ImageEvaluator.evaluation_metrics}
                    curr_zip_metrics_bilinear_demod = {k: 0.0 for k in ImageEvaluator.evaluation_metrics}

            average_metrics_bilinear = {k: v / total_frames for k, v in average_metrics_bilinear.items()}
            average_metrics_bilinear_demod = {k: v / total_frames for k, v in average_metrics_bilinear_demod.items()}
            # Add the metrics to the results
            results_bilinear.average[upscale_factor] = average_metrics_bilinear
            results_bilinear.per_frame_metrics[upscale_factor] = per_frame_metrics_bilinear
            results_bilinear.per_zip[upscale_factor] = zip_metrics_bilinear
            results_bilinear_demod.average[upscale_factor] = average_metrics_bilinear_demod
            results_bilinear_demod.per_frame_metrics[upscale_factor] = per_frame_metrics_bilinear_demod
            results_bilinear_demod.per_zip[upscale_factor] = zip_metrics_bilinear_demod


        return results_bilinear, results_bilinear_demod            
