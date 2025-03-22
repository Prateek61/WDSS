import numpy as np

from typing import Tuple, Dict, Any, Union

# Typedef
PatchWindow = Tuple[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[Tuple[int, int], Tuple[int, int]]]

class Patch:
    def __init__(
        self,
        high_resolution: Tuple[int, int],
        high_resolution_patch_size: int, # 0 for no patching 
        multi_patches_per_image: bool = False
    ):
        self.high_resolution = high_resolution
        self.high_resolution_patch_size = high_resolution_patch_size
        self.multi_patches_per_image = multi_patches_per_image
        self.multi_patches_per_image = multi_patches_per_image

        if (self.multi_patches_per_image):
            raise NotImplementedError("Multi patches per frame is not implemented yet")
        
        self._patches_per_frame: int = self._compute_patches_per_frame()

    @property
    def patches_per_frame(self) -> int:
        return self._patches_per_frame
    
    def get_patch_window(self, upscale_factor: float, patch_idx: int = 0) -> PatchWindow:
        if self.high_resolution_patch_size == 0:
            return self._full_image_patch_window(upscale_factor)
        elif not self.multi_patches_per_image:
            return self._random_patch_window(upscale_factor)
        else:
            raise NotImplementedError("Custom patch window not implemented for this configuration")

    def _full_image_patch_window(self, upscale_factor: float) -> PatchWindow:
        hr_h, hr_w = self.high_resolution
        lr_h, lr_w = int(hr_h / upscale_factor), int(hr_w / upscale_factor)
        return ((0, 0), (lr_h, lr_w)), ((0, 0), (hr_h, hr_w))

    def _compute_patches_per_frame(self) -> int:
        return 1
    
    def _random_patch_window(self, upscale_factor: float) -> PatchWindow:
        hr_h, hr_w = self.high_resolution
        patch_h, patch_w = self.high_resolution_patch_size, self.high_resolution_patch_size

        # Randomly sample the top left corner of the patch
        top_left_h = np.random.randint(0, hr_h - patch_h + 1)
        top_left_w = np.random.randint(0, hr_w - patch_w + 1)

        return self._patch_window_from_hr_patch((top_left_h, top_left_w), upscale_factor)

    def _patch_window_from_hr_patch(self, patch: Tuple[int, int], upscale_factor: float) -> PatchWindow:
        hr_patch_h, hr_patch_w = self.high_resolution_patch_size, self.high_resolution_patch_size
        lr_patch_h, lr_patch_w = int(hr_patch_h / upscale_factor), int(hr_patch_w / upscale_factor)
        hr_h, hr_w = self.high_resolution

        hr_patch_y_tl, hr_patch_x_tl = patch

        # Clamp the patch to fit within the image
        hr_patch_y_tl = int(min(max(0, hr_patch_y_tl), hr_h - hr_patch_h))
        hr_patch_x_tl = int(min(max(0, hr_patch_x_tl), hr_w - hr_patch_w))

        # Calculate the bottom right corner of the patch
        hr_patch_y_br = int(hr_patch_y_tl + hr_patch_h)
        hr_patch_x_br = int(hr_patch_x_tl + hr_patch_w)

        # Calculate the top left and bottom right corners of the low resolution patch
        lr_patch_y_tl = int(hr_patch_y_tl // upscale_factor)
        lr_patch_x_tl = int(hr_patch_x_tl // upscale_factor)
        lr_patch_y_br = int(lr_patch_y_tl + lr_patch_h)
        lr_patch_x_br = int(lr_patch_x_tl + lr_patch_w)

        # Return the patch window
        lr_window = ((lr_patch_y_tl, lr_patch_x_tl), (lr_patch_y_br, lr_patch_x_br))
        hr_window = ((hr_patch_y_tl, hr_patch_x_tl), (hr_patch_y_br, hr_patch_x_br))
        return lr_window, hr_window