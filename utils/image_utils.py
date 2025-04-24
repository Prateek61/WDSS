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
        res = F.interpolate(input, scale_factor=scale_factor, mode=mode, align_corners=False)
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
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        return image
    
    @staticmethod
    def save_exr_image_opencv(image: np.ndarray, image_path: str) -> None:
        """Save an image as an .exr file using OpenCV.

        Args:
            :attr:`image` (np.ndarray): Image to save. Shape (H, W, C).
            :attr:`image_path` (str): Path to save the .exr image.
        """

        # Convert the image to OpenCV format
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        # Save the image
        cv2.imwrite(image_path, image)

    @staticmethod
    def decode_exr_image_opencv(file_buffer: bytes) -> np.ndarray:
        # Decode the EXR image using OpenCV
        image = cv2.imdecode(np.frombuffer(file_buffer, np.uint8), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
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
                image = np.clip(image, 0, 255)
            else:
                # Clip the pixel values to [0, 1]
                image = np.clip(image, 0.0, 1.0)
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
                    image = np.clip(image, 0, 255)
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

    @staticmethod
    def tone_map(image: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
        """Apply gamma correction to the input image tensor.

        Returns:
            torch.Tensor: Tone-mapped image tensor.
        """

        # Apply gamma correction
        image = image ** (1.0 / gamma)
        return image.clamp(0, 1)
    
    @staticmethod
    def tone_de_map(image: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
        """Apply gamma correction to the input image tensor.

        Returns:
            torch.Tensor: Tone-mapped image tensor.
        """

        # Apply gamma correction
        image = image ** gamma
        return image.clamp(0, 1)
    
    @staticmethod
    def aces_tonemap(image: torch.Tensor, gain: float = 1.4) -> torch.Tensor:
        A = 2.51
        B = 0.03
        C = 2.43
        D = 0.59
        E = 0.14

        pre_tonemapping_transform = torch.Tensor([
            [0.575961650,  0.344143820,  0.079952030],
            [0.070806820,  0.827392350,  0.101774690],
            [0.028035252,  0.131523770,  0.840242300]
        ])

        post_tonemapping_transform = torch.Tensor([
            [1.666954300, -0.601741150, -0.065202855],
            [-0.106835220,  1.237778600, -0.130948950],
            [-0.004142626, -0.087411870,  1.091555000]
        ])

        exposed_pretonemapped_transform = gain * pre_tonemapping_transform
        
        # Move channels to the last
        image = image.permute(0, 2, 3, 1)

        image = image @ exposed_pretonemapped_transform.T

        image = (image * (A * image + B)) / (image * (C * image + D) + E).clamp(min=0.0, max=1.0)

        image = image @ post_tonemapping_transform.T

        return image.permute(0, 3, 1, 2).clamp(min=0.0, max=1.0)
    
    @staticmethod
    def save_tensor(image: torch.Tensor, path: str) -> None:
        """Save a tensor as an image file.
        """

        image_cv = ImageUtils.tensor_to_opencv_image(image)
        image_cv = (image_cv * 255).astype(np.uint8)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, image_cv)

    @staticmethod
    def stack_wavelet(wavelet: torch.Tensor) -> torch.Tensor:
        """Stack the wavelets for display.
        """
        batch_dim: bool = wavelet.dim() == 4
        if not batch_dim:
            # If batch dimension is present, remove it
            wavelet = wavelet.unsqueeze(0)
        
        b, c, h, w = wavelet.shape

        if c != 12:
            raise ValueError(f"Expected 12 channels, got {c} channels.")

        # Get the coefficients
        cA = wavelet[:, 0:3, :, :]
        cH = wavelet[:, 3:6, :, :]
        cV = wavelet[:, 6:9, :, :]
        cD = wavelet[:, 9:12, :, :]

        # Stack the wavelets
        # Final image will be of shape (3, 2H, 2W)
        wavelet_img = torch.zeros(b, 3, 2*h, 2*w)

        wavelet_img[:, :, :h, :w] = cA
        wavelet_img[:, :, :h, w:] = cV
        wavelet_img[:, :, h:, :w] = cD
        wavelet_img[:, :, h:, w:] = cH

        if not batch_dim:
            # If batch dimension was present, add it back
            wavelet_img = wavelet_img.squeeze(0)

        return wavelet_img

    @staticmethod
    def aces_tonemap_fast(image: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
        A = 2.51
        B = 0.03
        C = 2.43
        D = 0.59
        E = 0.14

        image = image * gain

        image = (image * (A * image + B)) / (image * (C * image + D) + E)

        return image.clamp(0.0, 1.0)


    @staticmethod 
    def _hable_tonemap_core(x: torch.Tensor) -> torch.Tensor:
        hA = 0.15
        hB = 0.50
        hC = 0.10
        hD = 0.20
        hE = 0.02
        hF = 0.30

        return ((x * (hA * x + hC * hB) + hD * hE) / (x * (hA * x + hB) + hD * hF)) - (hE / hF)
    
    @staticmethod
    def hable_tonemap(x: torch.Tensor) -> torch.Tensor:
        x = ImageUtils._hable_tonemap_core(x)
        hw = torch.as_tensor(11.2)
        white_scale = 1.0 / ImageUtils._hable_tonemap_core(hw)
        x = x * white_scale
        return x.clamp(0.0, 1.0)
    
    @staticmethod
    def tonemap(image: torch.Tensor) -> torch.Tensor:
        return ImageUtils.aces_tonemap_fast(image, gain=1.6)
