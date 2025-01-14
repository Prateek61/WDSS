import torch
import pywt
import ptwt
import matplotlib.pyplot as plt
import cv2 as cv


class WaveletProcessor:
    @staticmethod
    def wavelet_transform_image(image, wavelet='haar', level=1):
        """
        Apply wavelet transform to each color channel of the image on GPU.
        Returns RGB channel coefficients in the following order:
        [R_approx, G_approx, B_approx, R_horizontal, G_horizontal, B_horizontal,
        R_vertical, G_vertical, B_vertical, R_diagonal, G_diagonal, B_diagonal]
        """
        if len(image.shape) != 3 or image.shape[0] != 3:
            raise ValueError("Input image must have shape (3, H, W) for RGB format.")

        channels = torch.split(image, 1, dim=0)  # Split into R, G, B channels
        coefficients = []

        for channel in channels:
            coeffs = ptwt.wavedec2(channel, pywt.Wavelet(wavelet), level=level)
            
            
            coeffs = (coeffs[0].squeeze(0), tuple(c.squeeze(0) for c in coeffs[1]))
            
            coefficients.append(coeffs)

        coeff_shape = coefficients[0][1][0].shape
        channel_coeffs = torch.zeros((12, *coeff_shape), device=image.device)


        for i, coeffs in enumerate(coefficients):
            channel_coeffs[i] = coeffs[0]  # Approximation
            channel_coeffs[i + 3] = coeffs[1][0]  # Horizontal detail
            channel_coeffs[i + 6] = coeffs[1][1]  # Vertical detail
            channel_coeffs[i + 9] = coeffs[1][2]  # Diagonal detail


        return channel_coeffs

    @staticmethod
    def normalize_coefficients(coeffs):
        """
        Normalize wavelet coefficients to the range [0, 255] for visualization.
        """
        coeffs_min = coeffs.min()
        coeffs_max = coeffs.max()
        norm_coeffs = (coeffs - coeffs_min) / (coeffs_max - coeffs_min) * 255
        return norm_coeffs.to(torch.uint8)

    @staticmethod
    def reconstruct_image(coefficients, wavelet='haar'):
        """
        Reconstruct the original image from wavelet coefficients on GPU.
        """
        channels = []
        for i in range(3):  # R, G, B channels
            approx = coefficients[i].unsqueeze(0).unsqueeze(0)
            hor = coefficients[i + 3].unsqueeze(0).unsqueeze(0)
            ver = coefficients[i + 6].unsqueeze(0).unsqueeze(0)
            diag = coefficients[i + 9].unsqueeze(0).unsqueeze(0)
            coeffs = (approx, (hor, ver, diag))

            channel = ptwt.waverec2(coeffs, pywt.Wavelet(wavelet))[0, 0]
            channels.append(channel)

        reconstructed = torch.stack(channels, dim=0)
        return reconstructed.clamp(0, 255).float()

    @staticmethod
    def batch_wt(image_batch, wavelet='haar', level = 1):
        """
        Apply wavelet transform to a batch of images.
        """
        coeffs_batch = []
        for image in image_batch:
            coeffs = WaveletProcessor.wavelet_transform_image(image, wavelet, level)
            coeffs_batch.append(coeffs)
        return torch.stack(coeffs_batch, dim=0)

    def batch_iwt(coeffs_batch, wavelet='haar'):
        """
        Reconstruct a batch of images from wavelet coefficients.
        """
        images = []
        for coeffs in coeffs_batch:
            image = WaveletProcessor.reconstruct_image(coeffs, wavelet)
            images.append(image)
        return torch.stack(images, dim=0)


    @staticmethod
    def test_pipeline(image, wavelet='haar', level=1):
        """
        Test the full wavelet processing pipeline: transform, normalize, reconstruct, and evaluate.
        """
        # Perform wavelet transform
        wavelet_coeffs = WaveletProcessor.wavelet_transform_image(image, wavelet, level)

        # Reconstruct the image
        reconstructed_image = WaveletProcessor.reconstruct_image(wavelet_coeffs)

        # Calculate PSNR
        mse = torch.mean((image - reconstructed_image) ** 2)
        psnr = 10 * torch.log10(255**2 / mse)
        print(f"PSNR: {psnr.item():.2f} dB")


if __name__ == '__main__':
    # Load an image and apply wavelet transform
    image  = cv.imread('g.jpg')
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    print(image.shape)
    
    
    wt = WaveletProcessor.wavelet_transform_image(image, wavelet='haar', level=1)
    print(wt.shape)
    
    iwt = WaveletProcessor.reconstruct_image(wt, wavelet='haar')
    print(iwt.shape)
    
    WaveletProcessor.test_pipeline(image, wavelet='haar', level=1)
    
