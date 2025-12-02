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
from diffusion import Diffusion,  SR3UNet
from utils.save_checkpoint_diff import save_checkpoint
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random
from fvcore.nn import FlopCountAnalysis

# Alert messages
import alert

def main():

    # config = {
    #     'dataset': 'DIV2K',
    #     'img_size': (640, 640, 3),
    #     'timestep_embedding_dim': 256,
    #     'n_layers': 8,
    #     'hidden_dim': 32,
    #     'n_timesteps': 400,
    #     'train_batch_size': 128,
    #     'inference_batch_size': 64,
    #     'lr': 1e-4,
    #     'epochs': 1000,
    #     'seed': 42,
    # }
    config = {
        'dataset': 'DIV2K',
        'img_size': (640, 640, 3),
        'timestep_embedding_dim': 256,
        'n_layers': 8,
        'hidden_dim': 32,
        'n_timesteps': 1000,
        'train_batch_size': 128,
        'inference_batch_size': 64,
        'lr': 1e-4,
        'epochs': 1000,
        'seed': 42,
    }

    hidden_dims = [config['hidden_dim'] for _ in range(config['n_layers'])]
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # For GPU (CUDA) or CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess the dataset
    basic_transforms = T.Compose([
        T.ToTensor(),       # Converts [0, 255] to [0.0, 1.0]
        T.Lambda(lambda t: (t * 2) - 1)
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
        batch_size=4,
        scale=8,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=cpu_count(),
        pin_memory=True,
    )

    checkpoint = "ckpt_PSNR_18.5808.pth"
    checkpointPath = "super-resolution-model-tests/diffusion/training_checkpoints/"+checkpoint

    model = SR3UNet(in_channels = 3,
                     cond_channels = 3,
                     base_channels = config['hidden_dim'],
                     time_dim = config['timestep_embedding_dim']
                     ).to(device)

    state_dict = torch.load(checkpointPath, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])

    diffusion = Diffusion(model,  
                          timesteps=config['n_timesteps'],
                          device=device
                          )

    scale = 8

    model.eval()

    visualise_validation_set(val_loader, diffusion, model, device, scale)

def compute_mse(img1, img2):
    return F.mse_loss(img1, img2)

def compute_psnr(mse):
    mse_tensor = torch.tensor(mse) 
    return 10 * torch.log10(1 / mse_tensor)

def compute_flops(model, img, device='cuda'):
    flops = FlopCountAnalysis(model, img)
    total_flops = flops.total()

    print(f"Approximate FLOPs: {total_flops / 1e9:.2f} GFLOPs")

def visualise_validation_set(val_loader, diffusion, model, device, scale):
    image_dir = 'super-resolution-model-tests/test_vals/diffusion/generated_image'
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    imageChosen = None
    scale_factor = val_loader.dataset.scale
    imageChosenNum = [0, 50, 70]
    ssim_loss_fn = SSIMLoss().to(device)
    num_images_to_check = 8

    total_start = time.time()
    # Loop over the entire validation loader
    for idx, (lr_img_full, hr_img_full) in enumerate(val_loader):
        if (idx > num_images_to_check) and (idx not in imageChosenNum):
            continue
        start_time = time.time()
        lr_img_full = lr_img_full.to(device)
        hr_img_full = hr_img_full.to(device)
        
        with torch.no_grad():
            B, C, H, W = lr_img_full.shape
            sr_img = diffusion.sample(lr_img_full, shape=(B, C, H*scale, W*scale)).to(device).squeeze(0)
            
        hr_img_vis = hr_img_full.squeeze(0)
        hr_img_vis = (hr_img_vis.clamp(-1, 1) + 1) / 2
        sr_img_vis = (sr_img.clamp(-1, 1) + 1) / 2

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

        print(f'Image Build Time: {time.time() - start_time:.2f} sec, psnr: {psnr_sr}')

        # Save image
        if idx in imageChosenNum:
            image = (sr_img_vis.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            filename = f"{image_dir}/image_{idx}.png"
            Image.fromarray(image).save(filename)

    avg_psnr = total_psnr / count if count > 0 else 0.0
    avg_ssim = total_ssim / count
    avg_time = (time.time() - total_start) / count

    alert.send_notification(f"Diffusion test complete: Avg PSNR {avg_psnr:.2f}, Avg SSIM {avg_ssim:.2f}, Avg Time {avg_time:.2f} sec")

    compute_flops(model, torch.randn(1, 3, 64, 64).to(device))


    # Print Final PSNR value
    print(f"Diffusion test complete: Avg PSNR {avg_psnr:.2f}, Avg SSIM {avg_ssim:.2f}, Avg Time {avg_time:.2f} sec")


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