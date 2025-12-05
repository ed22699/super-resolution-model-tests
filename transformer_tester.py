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
from dataloader import GANDIV2KDataLoader
from utils.save_checkpoint_diff import save_checkpoint
from transformer import SwinIR
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random
from fvcore.nn import FlopCountAnalysis
from torchvision.transforms import v2
import lpips

# Alert messages
# import alert

def main():
    config = {
        'embed_dim': 128,
        'upscale': 8,
    }

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess the dataset
    basic_transforms_Hr = T.Compose([
        T.ToTensor(),
    ])
    basic_transforms_Lr = T.Compose([
        T.ToTensor(),  
    ])

    datasetRoot = "DIV2K"

    lr_path = datasetRoot+"/DIV2K_valid_LR_x8"
    hr_path = datasetRoot+"/DIV2K_valid_HR"

    val_dataset = GANDIV2KDataLoader(
        root_dir_lr=lr_path,
        root_dir_hr=hr_path,
        transformLr=basic_transforms_Lr,
        transformHr=basic_transforms_Hr,
        mode="val",
        batch_size=4,
        scale=config['upscale'],
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=cpu_count(),
        pin_memory=True,
    )


    model = SwinIR(in_ch = 3,
                     embed_dim=config['embed_dim'],
                     upscale = config['upscale'],
                     ).to(device)


    checkpoint = "ckpt_PSNR_21.9209.pth"
    checkpointPath = "super-resolution-model-tests/transformer/training_checkpoints/"+checkpoint

    state_dict = torch.load(checkpointPath, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])

    model.eval()

    visualise_validation_set(val_loader, model, device, config['upscale'])

def compute_psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    psnr = 10 * torch.log10((max_val ** 2) / mse)
    return psnr

def compute_flops(model, img, device='cuda'):
    flops = FlopCountAnalysis(model, img)
    total_flops = flops.total()

    print(f"Approximate FLOPs: {total_flops / 1e9:.2f} GFLOPs")

def visualise_validation_set(val_loader, model, device, scale):
    image_dir = 'super-resolution-model-tests/test_vals/transformer/16x/generated_image'
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    count = 0
    imageChosen = None
    scale_factor = val_loader.dataset.scale
    imageChosenNum = [0, 50, 70]

    # SSIM setup
    ssim_loss_fn = SSIMLoss().to(device)
    
    # LPIPS setup
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
    norm = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    total_start = time.time()

    # Loop over the entire validation loader
    for idx, (lr_img_full, hr_img_full) in enumerate(val_loader):
        
        _, _, H_lr, W_lr = lr_img_full.shape
        
        TILE_SIZE = 128
        OVERLAP = 32
        STRIDE = TILE_SIZE - OVERLAP

        # Create tensors to store the stitched image
        H_sr = H_lr * scale_factor
        W_sr = W_lr * scale_factor
        sr_img_stitched = torch.zeros(hr_img_full.shape[1], H_sr, W_sr, dtype=hr_img_full.dtype).to(device)
        stitch_counter = torch.zeros(hr_img_full.shape[1], H_sr, W_sr, dtype=hr_img_full.dtype).to(device)

        with torch.no_grad():
            for h_start in range(0, H_lr, STRIDE):
                for w_start in range(0, W_lr, STRIDE):
                    # Define LR patch coordinates
                    h_end = min(h_start + TILE_SIZE, H_lr)
                    w_end = min(w_start + TILE_SIZE, W_lr)
                    
                    # Adjust start
                    h_start_actual = max(0, h_end - TILE_SIZE)
                    w_start_actual = max(0, w_end - TILE_SIZE)

                    # Extract LR patch
                    lr_patch = lr_img_full[:, :, h_start_actual:h_end, w_start_actual:w_end].to(device)
                
                    # Generate SR patch
                    sr_patch = model(lr_patch).squeeze(0) # Keep on device for now

                    h_start_sr = h_start_actual * scale_factor
                    h_end_sr = h_end * scale_factor
                    w_start_sr = w_start_actual * scale_factor
                    w_end_sr = w_end * scale_factor

                    # Add patch to stiched image
                    sr_img_stitched[:, h_start_sr:h_end_sr, w_start_sr:w_end_sr] += sr_patch
                    stitch_counter[:, h_start_sr:h_end_sr, w_start_sr:w_end_sr] += 1.0

        sr_img_stitched /= stitch_counter

        sr_img_stitched = sr_img_stitched.cpu()
        hr_img_vis = hr_img_full.squeeze(0).clamp(0, 1)
        sr_img_vis = sr_img_stitched.clamp(0, 1)

        # Align sizing issues from patching
        H = min(hr_img_vis.shape[1], sr_img_vis.shape[1])
        W = min(hr_img_vis.shape[2], sr_img_vis.shape[2])
        hr_img_vis = hr_img_vis[:, :H, :W]
        sr_img_vis = sr_img_vis[:, :H, :W]

        # Calculate Metrics
        psnr_sr = compute_psnr(hr_img_vis, sr_img_vis)
        ssim_loss = ssim_loss_fn(sr_img_vis.unsqueeze(0), hr_img_vis.unsqueeze(0))
        total_ssim += (1- ssim_loss)

        ref_input = norm(hr_img) 
        pred_input = norm(sr_img)
        
        with torch.no_grad():
            lpips_score = lpips_loss_fn(pred_input, ref_input)
    
        # Accumulate metrics
        total_psnr += psnr_sr
        total_lpips += lpips_score.item()
        count += 1

        # Save image
        if idx in imageChosenNum:
            image = (sr_img_vis.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            filename = f"{image_dir}/image_{idx}.png"
            Image.fromarray(image).save(filename)
            

    avg_psnr = total_psnr / count if count > 0 else 0.0
    avg_ssim = total_ssim / count
    avg_lpips = total_lpips / count if count > 0 else 0.0
    avg_time = (time.time() - total_start) / count

    # alert.send_notification(f"Transformer test complete: Avg PSNR {avg_psnr:.2f}, Avg SSIM {avg_ssim:.2f}, Avg Time {avg_time:.2f} sec")

    compute_flops(model, torch.randn(1, 3, 64, 64).to(device))


    # Print Final PSNR value
    print(f"Transformer test complete: Avg PSNR {avg_psnr:.2f}, Avg SSIM {avg_ssim:.2f}, Avg LPIPS {avg_lpips:.2f}, Avg Time {avg_time:.2f} sec")


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



if __name__ == "__main__":
    main()