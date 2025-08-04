import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image
import cv2
import OpenEXR
import Imath
import io
import tempfile

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
        res = F.interpolate(input, scale_factor=scale_factor, mode=mode)
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
    def _map_imath_pixel_types(
        type: Imath.PixelType
    ) -> np.dtype:
        """
        Map Imath.PixelType to numpy dtype.

        Args:
            type (Imath.PixelType): Imath pixel type.

        Returns:
            np.dtype: Corresponding numpy dtype.
        """
        if Imath.PixelType(type) == Imath.PixelType(Imath.PixelType.HALF):
            return np.float16
        elif Imath.PixelType(type) == Imath.PixelType(Imath.PixelType.FLOAT):
            return np.float32
        else:
            raise ValueError(f"Unsupported PixelType: {type}. Supported types are: [HALF, FLOAT].")

    @staticmethod
    def save_exr_image_openexr(
        image: np.ndarray,
        image_path: str,
        precisions: List[Imath.PixelType] = None,
        compression: bool = False
    ) -> None:
        """
        Save a 4-channel image as an .exr file using OpenEXR 3.3+ File API.

        Args:
            image (np.ndarray): Input image of shape (H, W, 4).
            image_path (str): Path to save the .exr image.
            precisions (List[Imath.PixelType], optional): List of 4 Imath.PixelType values for each channel.
                                                        Defaults to [HALF, HALF, HALF, HALF].
        """
        h, w, c = image.shape
        assert c == 4, "Image must have 4 channels (e.g., RGBA)."

        if precisions is None:
            precisions = [Imath.PixelType.HALF] * 4
        assert len(precisions) == 4, "Must provide precision for each of the 4 channels."

        channel_names = ['R', 'G', 'B', 'A']

        # Prepare image channels (convert based on PixelType)
        channels = {}
        for i, name in enumerate(channel_names):
            pixel_type = precisions[i] 
            dtype = ImageUtils._map_imath_pixel_types(pixel_type)
            channels[name] = image[:, :, i].astype(dtype)

        # Build EXR header
        header = {
            "compression": OpenEXR.PIZ_COMPRESSION if compression else OpenEXR.NO_COMPRESSION,
            "type" : OpenEXR.scanlineimage,
        }

        # Write using the new OpenEXR.File API
        with OpenEXR.File(header, channels) as exr_file:
            exr_file.write(image_path)

    @staticmethod
    def decode_exr_image_openexr(file_buffer: bytes) -> np.ndarray:
        """Decode an EXR image using OpenEXR.

        Args:
            file_buffer (bytes): Buffer containing the EXR image data.

        Returns:
            np.ndarray: Decoded image. Shape (H, W, 4).
        """

        # Create a temporary file to store the EXR data
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            # Write the EXR data to the temporary file
            temp_file.write(file_buffer)
            temp_file.flush()

            return ImageUtils.load_exr_image_openexr(temp_file.name)


    def load_exr_image_openexr(file_path: str) -> np.ndarray:
        """Load an EXR image using OpenEXR.

        Args:
            file_path (str): Path to the EXR image.

        Returns:
            np.ndarray: Loaded image. Shape (H, W, 4).
        """

        with OpenEXR.File(file_path) as exr_file:
            return exr_file.channels()["RGBA"].pixels

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
