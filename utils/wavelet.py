import torch
import pywt
import ptwt
import matplotlib.pyplot as plt
import cv2 as cv
from .swt import swavedec2, swaverec2


class WaveletProcessor:
    WAVELET_TRANSFORM_TYPE = 'dwt' # 'dwt', 'swt
    WAVELET_TYPE = 'haar' # Default wavelet type, can be changed to any valid PyWavelets wavelet

    @staticmethod
    def _wt(image: torch.Tensor, wavelet: str = WAVELET_TYPE, level: int = 1) -> torch.Tensor:
        if WaveletProcessor.WAVELET_TRANSFORM_TYPE == 'dwt':
            return ptwt.wavedec2(image, pywt.Wavelet(wavelet), level=level)
        elif WaveletProcessor.WAVELET_TRANSFORM_TYPE == 'swt':
            return swavedec2(image, pywt.Wavelet(wavelet), level=level)
        else:
            raise ValueError("Unsupported wavelet transform type. Use 'dwt' or 'swt'.")
        
    @staticmethod
    def _iwt(coeffs: torch.Tensor, wavelet: str = WAVELET_TYPE) -> torch.Tensor:
        if WaveletProcessor.WAVELET_TRANSFORM_TYPE == 'dwt':
            return ptwt.waverec2(coeffs, pywt.Wavelet(wavelet))
        elif WaveletProcessor.WAVELET_TRANSFORM_TYPE == 'swt':
            return swaverec2(coeffs, pywt.Wavelet(wavelet))
        else:
            raise ValueError("Unsupported inverse wavelet transform type. Use 'dwt' or 'swt'.")

    @staticmethod
    def get_num_coefficients(level: int) -> int:
        """
        Get the number of output channels for a given decomposition level.
        Returns 3 (approx) + 9 * level (details per level: 3 bands * 3 RGB channels)
        """
        return 3 + 9 * level

    @staticmethod
    def wavelet_transform_image(image, wavelet=WAVELET_TYPE, level=1):
        """
        Apply multi-level wavelet transform to each color channel of the image on GPU.
        
        For level=L, returns RGB channel coefficients in the following order:
        [R_approx, G_approx, B_approx,                                          # 3 approx coeffs
         R_h_L, G_h_L, B_h_L, R_v_L, G_v_L, B_v_L, R_d_L, G_d_L, B_d_L,         # level L (coarsest) details
         R_h_L-1, G_h_L-1, B_h_L-1, R_v_L-1, G_v_L-1, B_v_L-1, R_d_L-1, ...     # level L-1 details
         ...
         R_h_1, G_h_1, B_h_1, R_v_1, G_v_1, B_v_1, R_d_1, G_d_1, B_d_1]         # level 1 (finest) details
        
        Total output channels = 3 + 9 * level
        
        Note: For DWT, multi-level decomposition produces coefficients of different sizes,
        which cannot be stacked into a single tensor. Use SWT for multi-level decomposition
        with stacked output, or use wavelet_transform_image_multilevel_dwt for DWT which
        returns a list structure.
        """
        if len(image.shape) != 3 or image.shape[0] != 3:
            raise ValueError("Input image must have shape (3, H, W) for RGB format.")

        # For DWT with level > 1, coefficients have different sizes - cannot stack
        if WaveletProcessor.WAVELET_TRANSFORM_TYPE == 'dwt' and level > 1:
            raise ValueError(
                "DWT with level > 1 produces coefficients of different sizes that cannot be stacked. "
                "Use SWT (set WAVELET_TRANSFORM_TYPE = 'swt') for multi-level decomposition with "
                "uniform coefficient sizes, or use wavelet_transform_image_multilevel_dwt() for DWT."
            )

        channels = torch.split(image, 1, dim=0)  # Split into R, G, B channels
        coefficients = []

        for channel in channels:
            coeffs = WaveletProcessor._wt(channel, wavelet, level)
            # coeffs structure: [approx, (h_L, v_L, d_L), (h_L-1, v_L-1, d_L-1), ..., (h_1, v_1, d_1)]
            # Squeeze out the channel dimension from each coefficient
            approx = coeffs[0].squeeze(0)
            details = [tuple(c.squeeze(0) for c in detail_tuple) for detail_tuple in coeffs[1:]]
            coefficients.append((approx, details))

        # For SWT, all coefficients have the same shape
        coeff_shape = coefficients[0][0].shape
        num_channels = WaveletProcessor.get_num_coefficients(level)
        channel_coeffs = torch.zeros((num_channels, *coeff_shape), device=image.device, dtype=image.dtype)

        # Fill approximation coefficients (indices 0, 1, 2 for R, G, B)
        for i, coeffs in enumerate(coefficients):
            channel_coeffs[i] = coeffs[0]  # Approximation

        # Fill detail coefficients for each level
        # Details are ordered from coarsest (level L) to finest (level 1)
        for lvl_idx in range(level):
            base_idx = 3 + lvl_idx * 9  # Starting index for this level's details
            for i, coeffs in enumerate(coefficients):
                detail_tuple = coeffs[1][lvl_idx]  # (horizontal, vertical, diagonal) for this level
                channel_coeffs[base_idx + i] = detail_tuple[0]      # Horizontal
                channel_coeffs[base_idx + 3 + i] = detail_tuple[1]  # Vertical
                channel_coeffs[base_idx + 6 + i] = detail_tuple[2]  # Diagonal

        return channel_coeffs

    @staticmethod
    def wavelet_transform_image_multilevel_dwt(image, wavelet=WAVELET_TYPE, level=1):
        """
        Apply multi-level DWT to each color channel of the image.
        Returns a list structure that preserves different coefficient sizes.
        
        Returns:
            tuple: (approx_tensor, [detail_tensors_per_level])
                - approx_tensor: shape (3, H_L, W_L) - RGB approximation at coarsest level
                - detail_tensors_per_level: list of tensors, each shape (9, H_l, W_l)
                  ordered from coarsest to finest level
                  Each level contains: [R_h, G_h, B_h, R_v, G_v, B_v, R_d, G_d, B_d]
        """
        if len(image.shape) != 3 or image.shape[0] != 3:
            raise ValueError("Input image must have shape (3, H, W) for RGB format.")

        channels = torch.split(image, 1, dim=0)  # Split into R, G, B channels
        coefficients = []

        for channel in channels:
            coeffs = ptwt.wavedec2(channel, pywt.Wavelet(wavelet), level=level)
            approx = coeffs[0].squeeze(0)
            details = [tuple(c.squeeze(0) for c in detail_tuple) for detail_tuple in coeffs[1:]]
            coefficients.append((approx, details))

        # Stack approximation coefficients
        approx_tensor = torch.stack([coeffs[0] for coeffs in coefficients], dim=0)

        # Stack detail coefficients per level
        detail_tensors = []
        for lvl_idx in range(level):
            lvl_details = []
            for i in range(3):  # R, G, B
                detail_tuple = coefficients[i][1][lvl_idx]
                lvl_details.extend([detail_tuple[0], detail_tuple[1], detail_tuple[2]])
            # Reorder to [R_h, G_h, B_h, R_v, G_v, B_v, R_d, G_d, B_d]
            reordered = []
            for band_idx in range(3):  # h, v, d
                for ch_idx in range(3):  # R, G, B
                    reordered.append(lvl_details[ch_idx * 3 + band_idx])
            detail_tensors.append(torch.stack(reordered, dim=0))

        return approx_tensor, detail_tensors

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
    def reconstruct_image(coefficients, wavelet=WAVELET_TYPE, level=1):
        """
        Reconstruct the original image from multi-level wavelet coefficients on GPU.
        
        Args:
            coefficients: Tensor of shape (3 + 9*level, H, W) containing:
                [R_approx, G_approx, B_approx,
                 R_h_L, G_h_L, B_h_L, R_v_L, G_v_L, B_v_L, R_d_L, G_d_L, B_d_L,  # level L
                 ...
                 R_h_1, G_h_1, B_h_1, R_v_1, G_v_1, B_v_1, R_d_1, G_d_1, B_d_1]  # level 1
            wavelet: Wavelet type to use for reconstruction
            level: Number of decomposition levels
        """
        # Validate input shape
        expected_channels = WaveletProcessor.get_num_coefficients(level)
        if coefficients.shape[0] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} channels for level={level}, "
                f"but got {coefficients.shape[0]} channels."
            )

        channels = []
        for i in range(3):  # R, G, B channels
            # Extract approximation for this channel
            approx = coefficients[i].unsqueeze(0).unsqueeze(0)
            
            # Build the coefficient structure for reconstruction
            # Format: [approx, (details_level_L), (details_level_L-1), ..., (details_level_1)]
            coeffs_list = [approx]
            
            for lvl_idx in range(level):
                base_idx = 3 + lvl_idx * 9
                hor = coefficients[base_idx + i].unsqueeze(0).unsqueeze(0)
                ver = coefficients[base_idx + 3 + i].unsqueeze(0).unsqueeze(0)
                diag = coefficients[base_idx + 6 + i].unsqueeze(0).unsqueeze(0)
                coeffs_list.append((hor, ver, diag))

            coeffs = tuple(coeffs_list)
            channel = WaveletProcessor._iwt(coeffs, wavelet)[0, 0]
            channels.append(channel)

        reconstructed = torch.stack(channels, dim=0)
        return reconstructed.float()

    @staticmethod
    def reconstruct_image_multilevel_dwt(approx_tensor, detail_tensors, wavelet=WAVELET_TYPE):
        """
        Reconstruct the original image from multi-level DWT coefficients.
        
        Args:
            approx_tensor: shape (3, H_L, W_L) - RGB approximation at coarsest level
            detail_tensors: list of tensors, each shape (9, H_l, W_l)
                ordered from coarsest to finest level
                Each level contains: [R_h, G_h, B_h, R_v, G_v, B_v, R_d, G_d, B_d]
            wavelet: Wavelet type to use for reconstruction
        """
        level = len(detail_tensors)
        channels = []
        
        for ch_idx in range(3):  # R, G, B channels
            # Extract approximation for this channel
            approx = approx_tensor[ch_idx].unsqueeze(0).unsqueeze(0)
            
            # Build coefficient structure
            coeffs_list = [approx]
            
            for lvl_idx in range(level):
                detail_tensor = detail_tensors[lvl_idx]
                hor = detail_tensor[ch_idx].unsqueeze(0).unsqueeze(0)          # R_h, G_h, or B_h
                ver = detail_tensor[3 + ch_idx].unsqueeze(0).unsqueeze(0)      # R_v, G_v, or B_v
                diag = detail_tensor[6 + ch_idx].unsqueeze(0).unsqueeze(0)     # R_d, G_d, or B_d
                coeffs_list.append((hor, ver, diag))

            coeffs = tuple(coeffs_list)
            channel = ptwt.waverec2(coeffs, pywt.Wavelet(wavelet))[0, 0]
            channels.append(channel)

        reconstructed = torch.stack(channels, dim=0)
        return reconstructed.float()

    @staticmethod
    def batch_wt(image_batch, wavelet=WAVELET_TYPE, level=1):
        """
        Apply multi-level wavelet transform to a batch of images.
        
        Args:
            image_batch: Tensor of shape (B, 3, H, W)
            wavelet: Wavelet type to use
            level: Number of decomposition levels
            
        Returns:
            Tensor of shape (B, 3 + 9*level, H', W') where H', W' depend on transform type
        """
        coeffs_batch = []
        for image in image_batch:
            coeffs = WaveletProcessor.wavelet_transform_image(image, wavelet, level)
            coeffs_batch.append(coeffs)
        return torch.stack(coeffs_batch, dim=0)

    @staticmethod
    def batch_iwt(coeffs_batch, wavelet=WAVELET_TYPE, level=1):
        """
        Reconstruct a batch of images from multi-level wavelet coefficients.
        
        Args:
            coeffs_batch: Tensor of shape (B, 3 + 9*level, H, W)
            wavelet: Wavelet type to use
            level: Number of decomposition levels
            
        Returns:
            Tensor of shape (B, 3, H', W')
        """
        images = []
        for coeffs in coeffs_batch:
            image = WaveletProcessor.reconstruct_image(coeffs, wavelet, level)
            images.append(image)
        return torch.stack(images, dim=0)


    @staticmethod
    def test_pipeline(image, wavelet=WAVELET_TYPE, level=1):
        """
        Test the full wavelet processing pipeline: transform, normalize, reconstruct, and evaluate.
        
        Args:
            image: Input image tensor of shape (3, H, W)
            wavelet: Wavelet type to use
            level: Number of decomposition levels
        """
        # Perform wavelet transform
        wavelet_coeffs = WaveletProcessor.wavelet_transform_image(image, wavelet, level)
        
        print(f"Wavelet transform type: {WaveletProcessor.WAVELET_TRANSFORM_TYPE}")
        print(f"Decomposition level: {level}")
        print(f"Input shape: {image.shape}")
        print(f"Coefficients shape: {wavelet_coeffs.shape}")
        print(f"Expected channels: {WaveletProcessor.get_num_coefficients(level)} (3 approx + {9*level} details)")

        # Reconstruct the image
        reconstructed_image = WaveletProcessor.reconstruct_image(wavelet_coeffs, wavelet, level)

        # Calculate PSNR
        mse = torch.mean((image - reconstructed_image) ** 2)
        psnr = 10 * torch.log10(255**2 / mse)
        print(f"PSNR: {psnr.item():.2f} dB")