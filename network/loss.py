import torch
import torch.nn.functional as F
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torchvision import models
from lpips import LPIPS as lpips


class ImageComparator:
    @staticmethod
    def psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
        """
        Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.
        """
        mse = F.mse_loss(img1, img2)
        psnr_value = 20 * torch.log10(max_val / torch.sqrt(mse))
        return psnr_value.item()

    @staticmethod
    def ssim(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0):
        """
        Compute SSIM between two images.
        Ensures images are in the expected format and specifies data_range.
        """
        # Ensure images are in numpy format and have shape (H, W, C)
        # Convert to numpy format and transpose to (H, W, C)
        img1_np = img1.permute(1, 2, 0).cpu().numpy()  # Shape: (720, 1280, 3)
        img2_np = img2.permute(1, 2, 0).cpu().numpy()        # Shape: (720, 1280, 3)


        # Compute SSIM with explicit win_size and channel_axis
        try:
            # Set an appropriate window size (7 is commonly used) and the data range
            win_size = 7  # Can be adjusted depending on the image size
            ssim_value, _ = ssim(
                img1_np,
                img2_np,
                full=True,
                multichannel=True,
                data_range=img1_np.max() - img1_np.min(),  # Define the data range based on the image
                win_size=win_size,
                channel_axis=-1  # Assuming the color channels are the last axis (H, W, C)
            )
            # print(f"SSIM Value: {ssim_value:.4f}")
            return ssim_value
        except Exception as e:
            print(f"Error in SSIM computation: {e}")
            
            
    @staticmethod
    def lpips(img1: torch.Tensor, img2: torch.Tensor, device: str = 'cpu') -> float:
        """
        Compute the Learned Perceptual Image Patch Similarity (LPIPS) between two images.
        """
        lpips_model = lpips(net='vgg').to(device)
        img1_resized = F.interpolate(img1.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)
        img2_resized = F.interpolate(img2.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)
        lpips_value = lpips_model(img1_resized, img2_resized)
        return lpips_value.item()

    @staticmethod
    def get_vgg_features(image: torch.Tensor, feature_extractor, selected_layers: list):
        """
        Extract features from a VGG model at specific layers.
        """
        features = []
        x = image.unsqueeze(0)  # Add batch dimension
        for idx, layer in enumerate(feature_extractor):
            x = layer(x)
            if idx in selected_layers:
                features.append(x)
        return features

    @staticmethod
    def compare_images(img1: torch.Tensor, img2: torch.Tensor, label: str = "", device: str = 'cpu'):
        """
        Compare two images using PSNR, SSIM, and LPIPS metrics, and optionally include a label.
        """
        # Load VGG16 feature extractor
        feature_extractor = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        selected_layers = [2, 7, 12, 21, 30]

        for param in feature_extractor.parameters():
            param.requires_grad = False
        feature_extractor = feature_extractor.to(device)

        # Extract features from both images using VGG
        img1_features = ImageComparator.get_vgg_features(img1, feature_extractor, selected_layers)
        img2_features = ImageComparator.get_vgg_features(img2, feature_extractor, selected_layers)

        # Compute metrics
        psnr_value = ImageComparator.psnr(img1, img2, max_val=1.0)
        ssim_value = ImageComparator.ssim(img1, img2)
        lpips_value = ImageComparator.lpips(img1, img2, device=device)

        # Print the metrics
        print(f"Comparison results for {label if label else 'unnamed images'}:")
        print(f"PSNR: {psnr_value:.2f}")
        print(f"SSIM: {ssim_value:.4f}")
        print(f"LPIPS: {lpips_value:.4f}")

        # Return the extracted features and metrics
        return {
            "psnr": psnr_value,
            "ssim": ssim_value,
            "lpips": lpips_value,
            "img1_features": img1_features,
            "img2_features": img2_features,
        }


# Usage Example
if __name__ == "__main__":
    # Example input images (assuming they're already loaded as torch tensors)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load and preprocess the images
    def preprocess_image(img: np.ndarray, device: str):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img = torch.tensor(img).permute(2, 0, 1).float()  # Shape: (C, H, W)
        return img.to(device)

    image1 = preprocess_image(cv2.imread("image1.png"), device)
    image2 = preprocess_image(cv2.imread("image2.png"), device)

    # Compare images
    results = ImageComparator.compare_images(image1, image2, label="Test Images", device=device)
