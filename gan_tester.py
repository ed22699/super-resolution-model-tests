import numpy as np
from multiprocessing import cpu_count
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import time
from PIL import Image
from gan_dataloader import GANDIV2KDataLoader
from gan import Generator, Discriminator, VGGFeatureExtractor
from utils.save_checkpoint_diff import save_checkpoint
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random
from fvcore.nn import FlopCountAnalysis

# Alert messages
import alert

def main():
    # For GPU (CUDA) or CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess the dataset
    basic_transforms = T.Compose([
        T.ToTensor(),       # Converts [0, 255] to [0.0, 1.0]
    ])

    datasetRoot = "DIV2K"

    lr_path = datasetRoot+"/DIV2K_valid_LR_x8"
    hr_path = datasetRoot+"/DIV2K_valid_HR"

    val_dataset = GANDIV2KDataLoader(
        root_dir_lr=lr_path,
        root_dir_hr=hr_path,
        transformLr=basic_transforms,
        transformHr=basic_transforms,
        mode="val",
        batch_size=64,
        scale=8,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=cpu_count(),
        pin_memory=True,
    )


    checkpoint = "ckpt_PSNR_22.4838.pth"
    checkpointPath = "super-resolution-model-tests/gan/training_checkpoints/"+checkpoint

    # Define the Binary Cross Entropy loss function
    loss_func = nn.BCEWithLogitsLoss()
    input_dim = 3
    num_epoch = 500

    gan_G = Generator(input_dim).to(device)

    state_dict = torch.load(checkpointPath, map_location=device)
    gan_G.load_state_dict(state_dict['generator_state_dict'])

    scale = 8

    gan_G.eval()

    visualise_validation_set(val_loader, gan_G, device, scale)

def compute_mse(img1, img2):
    return F.mse_loss(img1, img2)

def compute_psnr(mse):
    mse_tensor = torch.tensor(mse) 
    return 10 * torch.log10(1 / mse_tensor)

def compute_flops(model, img, device='cuda'):
    flops = FlopCountAnalysis(model, img)
    total_flops = flops.total()

    print(f"Approximate FLOPs: {total_flops / 1e9:.2f} GFLOPs")

def visualise_validation_set(val_loader, gan_G, device, scale):
    image_dir = 'super-resolution-model-tests/test_vals/GAN/generated_image'

    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    imageChosenNum = [0, 50, 70]
    imageChosen = None
    scale_factor = val_loader.dataset.scale
    ssim_loss_fn = SSIMLoss().to(device)
    total_start = time.time()

    # Loop over the entire validation loader
    for idx, (lr_img_full, hr_img_full) in enumerate(val_loader):
        
        _, _, H_lr, W_lr = lr_img_full.shape
        
        TILE_SIZE = 128
        OVERLAP = 32  # Define overlap amount (e.g., 25% of tile size)
        STRIDE = TILE_SIZE - OVERLAP

        # Create tensors to store the stitched image and a counter
        H_sr = H_lr * scale_factor
        W_sr = W_lr * scale_factor
        sr_img_stitched = torch.zeros(hr_img_full.shape[1], H_sr, W_sr, dtype=hr_img_full.dtype).to(device)
        stitch_counter = torch.zeros(hr_img_full.shape[1], H_sr, W_sr, dtype=hr_img_full.dtype).to(device)

        with torch.no_grad():
            # Loop with STRIDE instead of TILE_SIZE
            for h_start in range(0, H_lr, STRIDE):
                for w_start in range(0, W_lr, STRIDE):
                    # Define LR patch coordinates
                    h_end = min(h_start + TILE_SIZE, H_lr)
                    w_end = min(w_start + TILE_SIZE, W_lr)
                    
                    # Adjust start if we are at the end to ensure full tile size
                    h_start_actual = max(0, h_end - TILE_SIZE)
                    w_start_actual = max(0, w_end - TILE_SIZE)

                    # Extract LR patch
                    lr_patch = lr_img_full[:, :, h_start_actual:h_end, w_start_actual:w_end].to(device)
                
                    # Generate SR patch
                    sr_patch = gan_G(lr_patch).squeeze(0) # Keep on device for now

                    # Define SR patch coordinates
                    h_start_sr = h_start_actual * scale_factor
                    h_end_sr = h_end * scale_factor
                    w_start_sr = w_start_actual * scale_factor
                    w_end_sr = w_end * scale_factor

                    # Add patch to stiched image and increment counter
                    sr_img_stitched[:, h_start_sr:h_end_sr, w_start_sr:w_end_sr] += sr_patch
                    stitch_counter[:, h_start_sr:h_end_sr, w_start_sr:w_end_sr] += 1.0

        # Divide by counter to average overlapping regions
        sr_img_stitched /= stitch_counter

        # Move back to CPU for final processing
        sr_img_stitched = sr_img_stitched.cpu()
        hr_img_vis = hr_img_full.squeeze(0).clamp(0, 1)
        sr_img_vis = sr_img_stitched.clamp(0, 1)

        # Align sizing issues from patching
        H = min(hr_img_vis.shape[1], sr_img_vis.shape[1])
        W = min(hr_img_vis.shape[2], sr_img_vis.shape[2])
        hr_img_vis = hr_img_vis[:, :H, :W]
        sr_img_vis = sr_img_vis[:, :H, :W]

        # Calculate PSNR for this image
        mse_sr = compute_mse(hr_img_vis, sr_img_vis).item()
        psnr_sr = compute_psnr(mse_sr).item()
        sim_loss = ssim_loss_fn(sr_img_vis.unsqueeze(0), hr_img_vis.unsqueeze(0))
        total_ssim += (1 - sim_loss)
    
        # Accumulate PSNR
        total_psnr += psnr_sr
        count += 1

        # Grid
        # Save image
        if idx in imageChosenNum:
            image = (sr_img_vis.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            filename = f"{image_dir}/image_{idx}.png"
            Image.fromarray(image).save(filename)

    avg_psnr = total_psnr / count if count > 0 else 0.0
    avg_ssim = total_ssim / count
    avg_time = (time.time() - total_start) / count

    alert.send_notification(f"GAN test complete: Avg PSNR {avg_psnr:.2f}, Avg SSIM {avg_ssim:.2f}, Avg Time {avg_time:.2f} sec")

    compute_flops(gan_G, torch.randn(1, 3, 64, 64).to(device))


    # Print Final PSNR value
    print(f"GAN test complete: Avg PSNR {avg_psnr:.2f}, Avg SSIM {avg_ssim:.2f}, Avg Time {avg_time:.2f} sec")


def gaussian_window(window_size, sigma, channels):
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()

    # 2D gaussian
    g2d = g[:, None] * g[None, :]
    g2d = g2d / g2d.sum()

    # shape: (C, 1, k, k)
    window = g2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.register_buffer("window", None)  # will initialize later

    def forward(self, img1, img2):
        B, C, H, W = img1.shape

        # Create gaussian window on first use or when channels change
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

        return 1 - ssim     # loss



if __name__ == "__main__":
    main()