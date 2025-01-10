import torch
import torch.nn.functional as F
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torchvision import models, transforms
from lpips import LPIPS as lpips

def psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    mse = F.mse_loss(img1, img2)
    psnr_value = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr_value.item()

def load_image(image_path, device='cpu'):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img = torch.tensor(img).permute(2, 0, 1).float()  # Shape: (C, H, W)
    return img.to(device)

def get_vgg_features(image, feature_extractor, selected_layers):
    """
    Extracts features from the VGG model at specific layers.

    Args:
        image (torch.Tensor): The input image.
        feature_extractor (torch.nn.Module): The VGG feature extractor.
        selected_layers (list): List of selected layer indices.

    Returns:
        list: Extracted features from selected layers.
    """
    features = []
    x = image.unsqueeze(0)  # Add batch dimension
    for idx, layer in enumerate(feature_extractor):
        x = layer(x)
        if idx in selected_layers:
            features.append(x)
    return features

def compare_images(img1_path, img2_path, device='cpu'):
    img1 = load_image(img1_path, device=device)
    img2 = load_image(img2_path, device=device)

    # Load VGG16 feature extractor
    feature_extractor = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
    selected_layers = [2, 7, 12, 21, 30]

    for param in feature_extractor.parameters():
        param.requires_grad = False
    feature_extractor = feature_extractor.to(device)

    # Extract features from both images using VGG
    img1_features = get_vgg_features(img1, feature_extractor, selected_layers)
    img2_features = get_vgg_features(img2, feature_extractor, selected_layers)

    # Compute PSNR
    psnr_value = psnr(img1, img2, max_val=1.0)

    # Compute SSIM
    img1_np = img1.permute(1, 2, 0).cpu().numpy()
    img2_np = img2.permute(1, 2, 0).cpu().numpy()
    ssim_value, _ = ssim(img1_np, img2_np, full=True, multichannel=True)

    # Compute LPIPS
    lpips_model = lpips.LPIPS(net='vgg').to(device)  # Load the LPIPS model (VGG variant)
    img1_resized = F.interpolate(img1.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)
    img2_resized = F.interpolate(img2.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)
    lpips_value = lpips_model(img1_resized, img2_resized)

    # Print the metrics
    print(f"Comparison results:")
    print(f"PSNR: {psnr_value:.2f}")
    print(f"SSIM: {ssim_value:.4f}")
    print(f"LPIPS: {lpips_value.item():.4f}")

    # Optionally, return the extracted features for further inspection
    return img1_features, img2_features

