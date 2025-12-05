
import numpy as np
from multiprocessing import cpu_count
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image
from gan_dataloader import GANDIV2KDataLoader
import time
import lpips
import os

def gaussian_window(window_size, sigma, channels):
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    g2d = g[:, None] * g[None, :]
    g2d = g2d / g2d.sum()
    window = g2d.expand(channels, 1, window_size, window_size).contiguous()
    return window

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.register_buffer("window", None)

    def forward(self, img1, img2):
        B, C, H, W = img1.shape
        if self.window is None or self.window.size(0) != C:
            window = gaussian_window(self.window_size, self.sigma, C)
            self.window = window.to(img1.device).to(img1.dtype)
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=C)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=C)
        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size//2, groups=C) - mu1.pow(2)
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size//2, groups=C) - mu2.pow(2)
        sigma12   = F.conv2d(img1 * img2, self.window, padding=self.window_size//2, groups=C) - mu1 * mu2
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1.pow(2) + mu2.pow(2) + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim = ssim_map.mean()
        return 1 - ssim

def compute_psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return 100
    psnr = 10 * torch.log10((max_val ** 2) / mse)
    return psnr

def main():
    # Configuration
    config = {
        'scale': 8,
        'seed': 42,
    }

    # For GPU (CUDA) or CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    basic_transforms = T.Compose([
        T.ToTensor(),       
    ])

    # Load Dataset 
    datasetRoot = "DIV2K"
    lr_path = datasetRoot+"/DIV2K_valid_LR_x8"
    hr_path = datasetRoot+"/DIV2K_valid_HR"

    val_dataset = GANDIV2KDataLoader(
        root_dir_lr=lr_path,
        root_dir_hr=hr_path,
        transformLr=basic_transforms,
        transformHr=basic_transforms,
        mode="val",
        batch_size=1,
        scale=config['scale'],
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=cpu_count(),
        pin_memory=True,
    )

    evaluate_bicubic(val_loader, device, config['scale'])

def evaluate_bicubic(val_loader, device, scale):
    image_dir = 'super-resolution-model-tests/test_vals/bicubic/generated_image'
    os.makedirs(image_dir, exist_ok=True)
    
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    count = 0
    
    # Images to save
    imageChosenNum = [0, 50, 70]
    
    # SSIM setup
    ssim_loss_fn = SSIMLoss().to(device)
    
    # LPIPS setup
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
    norm = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    total_start = time.time()

    for idx, (lr_img, hr_img) in enumerate(val_loader):
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)
        
        target_h, target_w = hr_img.shape[-2], hr_img.shape[-1]
        
        # Upsample
        sr_img = F.interpolate(lr_img, size=(target_h, target_w), mode='bicubic', align_corners=False)
        
        sr_img = sr_img.clamp(0, 1)
        hr_img = hr_img.clamp(0, 1)

        # Calculate Metrics
        psnr_val = compute_psnr(hr_img, sr_img)
        
        sim_loss = ssim_loss_fn(sr_img, hr_img)
        ssim_val = 1 - sim_loss.item()

        ref_input = norm(hr_img) 
        pred_input = norm(sr_img)
        
        with torch.no_grad():
            lpips_score = lpips_loss_fn(pred_input, ref_input)
        
        
        # Add metrics to total
        total_lpips += lpips_score.item()
        total_psnr += psnr_val.item()
        total_ssim += ssim_val
        count += 1

        # Save image
        if idx in imageChosenNum:
            image = (sr_img.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            filename = f"{image_dir}/image_{idx}.png"
            Image.fromarray(image).save(filename)

    avg_psnr = total_psnr / count if count > 0 else 0.0
    avg_ssim = total_ssim / count if count > 0 else 0.0
    avg_lpips = total_lpips / count if count > 0 else 0.0
    avg_time = (time.time() - total_start) / count

    print(f"Avg PSNR: {avg_psnr:.2f} dB")
    print(f"Avg SSIM: {avg_ssim:.4f}")
    print(f"Avg LPIPS: {avg_lpips:.4f}")
    print(f"Avg Time: {avg_time:.4f} sec")

if __name__ == "__main__":
    main()