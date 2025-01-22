import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image
import cv2

from typing import List

class ImageUtils:
    """Static class containing utility functions for image processing
    """

    @staticmethod
    def upsample(input: torch.Tensor, scale_factor: float, mode: str = 'bilinear') -> torch.Tensor:
        """Upsample the input image tensor by a given scale factor.

        Args:
            :attr:`input` (torch.Tensor): Input 4D image tensor of shape (B, C, H, W).
            :attr:`scale_factor` (float): Scale factor for upsampling.
            :attr:`mode` (str): The upsampling algorithm. Default: 'bilinear'. Other options: 'nearest', 'bicubic', 'area'.

        Returns:
            torch.Tensor: Upsampled image tensor of shape (B, C, H', W').\n
            H' = H * scale_factor, W' = W * scale_factor.
        """

        # Check if the input is an instance of torch.Tensor
        if not isinstance(input, torch.Tensor):
            assert False, "Input must be a torch.Tensor"
        
        # Check if the input is a 4D tensor
        if input.dim() != 4:
            assert False, "Input must be a 4D tensor"
            
        # Perform upsampling
        res = F.interpolate(input, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        return res

    
    @staticmethod
    def image_to_tensor(image: Image.Image) -> torch.Tensor:
        """Convert a PIL image to a torch tensor.

        Returns:
            torch.Tensor: Output tensor. In format (1, C, H, W).
        """

        # Convert the PIL image to a tensor
        tensor = ToTensor()(image)
        # Add a batch dimension
        tensor = tensor.unsqueeze(0)
        return tensor


    @staticmethod
    def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
        """Convert a torch tensor to a PIL image.

        Args:
            :attr:`tensor` (torch.Tensor): Input tensor. In format (1, C, H, W) or (C, H, W).
        """

        # Check if the tensor is a 4D tensor (batch dimension)
        # If yes, remove the batch dimension
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        # Convert the tensor to a PIL image
        image = ToPILImage()(tensor)
        return image


    @staticmethod
    def opencv_image_to_tensor(image: np.ndarray) -> torch.Tensor:
        """Convert an OpenCV image to a torch tensor.

        Args:
            :attr:`image` (np.ndarray): Input OpenCV image. In format (H, W, C).

        Returns:
            torch.Tensor: Output tensor. In format (1, C, H, W).
        """

        # Convert the OpenCV image to a tensor
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        return tensor


    @staticmethod
    def tensor_to_opencv_image(tensor: torch.Tensor) -> np.ndarray:
        """Convert a torch tensor to an OpenCV image.

        Args:
            :attr:`tensor` (torch.Tensor): Input tensor. In format (1, C, H, W) or (C, H, W).

        Returns:
            np.ndarray: Output OpenCV image. In format (H, W, C).
        """

        # Check if the tensor is a 4D tensor (batch dimension)
        # If yes, remove the batch dimension
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        elif tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        # Convert the tensor to an OpenCV image
        image = tensor.permute(1, 2, 0).numpy()
        return image


    @staticmethod
    def load_exr_image_opencv(image_path: str) -> np.ndarray:
        """Load an .exr image using OpenCV.

        Args:
            :attr:`image_path` (str): Path to the .exr image.

        Returns:
            np.ndarray: Loaded image. Shape (H, W, C).
        """

        # Load the image using OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @staticmethod
    def decode_exr_image_opencv(file_buffer: bytes) -> np.ndarray:
        # Decode the EXR image using OpenCV
        image = cv2.imdecode(np.frombuffer(file_buffer, np.uint8), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @staticmethod
    def display_image(image: Image.Image | torch.Tensor | np.ndarray, title: str = "", normalize: bool = True) -> None:
        """Display an image using matplotlib.
        """

        # Check if the input is a tensor
        if isinstance(image, torch.Tensor):
            image = ImageUtils.tensor_to_opencv_image(image)
        # Number of channels in the image
        channels = 3
        # Check if the input is numpy array
        if isinstance(image, np.ndarray):
            # Normalize the image
            if normalize:
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                # Clip the pixel values to [0, 1]
                image = np.clip(image, 0, 1)
            try:
                channels = image.shape[2]
            except:
                channels = 1

        # Display the image
        if channels == 1: # If single channel, display in grayscale
            plt.imshow(image, cmap='gray')
        else: # Else display in color
            if channels == 2:
                # Add an extra B channel with zeros
                image = cv2.merge((image, np.zeros_like(image[:,:,0])))
            plt.imshow(image)

        plt.title(title)
        plt.axis('off')
        plt.show()


    @staticmethod
    def display_images(images: List[torch.Tensor | Image.Image | np.ndarray], titles: List[str] = [], normalize: bool = True) -> None:
        """Display a list of images using matplotlib.
        """

        # Display the images
        fig, axes = plt.subplots(1, len(images), figsize=(20, 10))
        if not titles:
            titles = [f"Image {i+1}" for i in range(len(images))]
        for i, (image, title) in enumerate(zip(images, titles)):
            channels = 3
            # Check if the input a tensor
            if isinstance(image, torch.Tensor):
                image = ImageUtils.tensor_to_opencv_image(image)
            # Check if the input is numpy array
            if isinstance(image, np.ndarray):
                # Normalize the image
                if normalize:
                    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                else:
                    # Clip the pixel values to [0, 1]
                    image = np.clip(image, 0, 1)
                try:
                    channels = image.shape[2]
                except:
                    channels = 1

            if channels == 1:
                axes[i].imshow(image, cmap='gray')
            else:
                if channels == 2:
                    # Add an extra B channel with zeros
                    image = cv2.merge((image, np.zeros_like(image[:,:,0])))
                axes[i].imshow(image)

            axes[i].set_title(title)
            axes[i].axis('off')
        plt.show()