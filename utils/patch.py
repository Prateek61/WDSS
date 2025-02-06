import torch
import numpy as np

from typing import Tuple, Dict, Any, Union

# Perform a typedef
PatchWindow = Tuple[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[Tuple[int, int], Tuple[int, int]]]

class Patch:
    def __init__(self, low_resolution: Tuple[int, int], upscale_factor: float, patch_size: int = 0, multi_patches_per_image: bool = False):
        self.low_resolution = low_resolution
        self.upscale_factor = upscale_factor
        self.patch_size = patch_size
        self.multi_patches_per_image = multi_patches_per_image

        self._patches_per_frame: int = 1
        self._custom: bool = False

        if self.low_resolution == (360, 640) and self.patch_size == 256 and multi_patches_per_image == True:
            self._custom = True
            self._patches_per_frame = 5
        elif self.multi_patches_per_image == True:
            raise NotImplementedError("Custom patch window not implemented for this configuration")


    @property
    def patches_per_frame(self) -> int:
        return self._patches_per_frame
    
    def get_patch_window(self, patch_idx: int = 0) -> PatchWindow:
        if self.patch_size == 0:
            return self._full_image_patch_window()
        elif self._custom:
            return self._custom_patch_window(patch_idx)
        else:
            return self._random_patch_window()


    def _full_image_patch_window(self) -> PatchWindow:
        return ((0, 0), self.low_resolution), ((0, 0), (self.low_resolution[0] * self.upscale_factor, self.low_resolution[1] * self.upscale_factor))

    def _random_patch_window(self) -> PatchWindow:
        lr_h, lr_w = self.low_resolution
        patch_h, patch_w = self.patch_size, self.patch_size

        # Randomly sample the top left corner of the patch
        top_left_h = np.random.randint(0, lr_h - patch_h + 1)
        top_left_w = np.random.randint(0, lr_w - patch_w + 1)

        # Calculate the bottom right corner of the patch
        bottom_right_h = top_left_h + patch_h
        bottom_right_w = top_left_w + patch_w

        # Calculate the top left and bottom right corners of the high resolution patch
        upscale_factor = self.upscale_factor
        upscale_patch_h = patch_h * upscale_factor
        upscale_patch_w = patch_w * upscale_factor
        
        upscale_top_left_h = top_left_h * upscale_factor
        upscale_top_left_w = top_left_w * upscale_factor

        upscale_bottom_right_h = upscale_top_left_h + upscale_patch_h
        upscale_bottom_right_w = upscale_top_left_w + upscale_patch_w

        return ((top_left_h, top_left_w), (bottom_right_h, bottom_right_w)), ((upscale_top_left_h, upscale_top_left_w), (upscale_bottom_right_h, upscale_bottom_right_w))
    

    def _custom_patch_window(self, patch_idx: int) -> PatchWindow:

        lr_h, lr_w = 360, 640
        patch_h, patch_w = 256, 256

        upscale_factor = self.upscale_factor

        h_window = lr_h - patch_h
        w_window = lr_w - patch_w

        if patch_idx == 0:
            patch_y, patch_x = h_window * 0.1, w_window * 0.1
        elif patch_idx == 1:
            patch_y, patch_x = h_window * 0.1, (lr_w - patch_w) - w_window * 0.1
        elif patch_idx == 2:
            patch_y, patch_x = (lr_h - patch_h) - h_window * 0.1, w_window * 0.1
        elif patch_idx == 3:
            patch_y, patch_x = (lr_h - patch_h) - h_window * 0.1, (lr_w - patch_w) - w_window * 0.1
        elif patch_idx == 4:
            patch_y, patch_x = h_window * 0.5, w_window * 0.5
        else:
            raise ValueError(f"Invalid patch index {patch_idx}")
        
        # Patch can move 30% of the lr_h and lr_w in the y and x direction
        patch_y += np.random.randint(-int(h_window * 0.15), int(h_window * 0.15))
        patch_x += np.random.randint(-int(w_window * 0.15), int(w_window * 0.15))

        # Clip the patch to the image boundaries
        patch_y = int(np.clip(patch_y, 0, lr_h - patch_h))
        patch_x = int(np.clip(patch_x, 0, lr_w - patch_w))

        hr_patch_y_tl = patch_y * upscale_factor
        hr_patch_x_tl = patch_x * upscale_factor
        hr_patch_y_br = hr_patch_y_tl + patch_h * upscale_factor
        hr_patch_x_br = hr_patch_x_tl + patch_w * upscale_factor

        lr_window = ((patch_y, patch_x), (patch_y + patch_h, patch_x + patch_w))
        hr_window = ((hr_patch_y_tl, hr_patch_x_tl), (hr_patch_y_br, hr_patch_x_br))
        return lr_window, hr_window
    