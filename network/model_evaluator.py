import torch
from .models.ModelBase import ModelBase
from utils.image_utils import ImageUtils
from .losses import ImageEvaluator
from .dataset import WDSSDatasetBase

class ModelEvaluator:
    @staticmethod
    def evaluate(model: ModelBase, data_loader: WDSSDatasetBase, image_no: int, device: torch.device):
        model.eval()
        with torch.no_grad():
            frame = data_loader.__getitem__(image_no)
            hr = frame['HR'].to(device).unsqueeze(0)
            lr: torch.Tensor = frame['LR'].to(device).unsqueeze(0)
            gb = frame['GB'].to(device).unsqueeze(0)
            temporal = frame['TEMPORAL'].to(device).unsqueeze(0)

            wavelet, img = model.forward(lr, gb, temporal, 2.0)

        bilinear_upsampled_lr = ImageUtils.upsample(lr, 2)

        mse = ImageEvaluator.mse(img, hr)
        psnr = ImageEvaluator.psnr(img, hr)
        ssim = ImageEvaluator.ssim(img, hr)
        lpips = ImageEvaluator.lpips(img, hr)

        mse_bilinear = ImageEvaluator.mse(bilinear_upsampled_lr, hr)
        psnr_bilinear = ImageEvaluator.psnr(bilinear_upsampled_lr, hr)
        ssim_bilinear = ImageEvaluator.ssim(bilinear_upsampled_lr, hr)
        lpips_bilinear = ImageEvaluator.lpips(bilinear_upsampled_lr, hr)

        print(f"MSE: Model: {mse.item()}, Bilinear: {mse_bilinear.item()}")
        print(f"PSNR: Model: {psnr.item()}, Bilinear: {psnr_bilinear.item()}")
        print(f"SSIM: Model: {ssim.item()}, Bilinear: {ssim_bilinear.item()}")
        print(f"LPIPS: Model: {lpips.item()}, Bilinear: {lpips_bilinear.item()}")

        ImageUtils.display_images([lr.detach().cpu(), hr.detach().cpu()], ["LR", "HR"])
        ImageUtils.display_images([img.detach().cpu(), bilinear_upsampled_lr.detach().cpu()], ["Model", "Bilinear"])
        