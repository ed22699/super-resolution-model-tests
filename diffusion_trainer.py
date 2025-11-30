import numpy as np
from multiprocessing import cpu_count
import sys
import os
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

# Alert messages
import alert


def main():

    config = {
        'dataset': 'DIV2K',
        'img_size': (640, 640, 3),
        'timestep_embedding_dim': 256,
        'n_layers': 8,
        'hidden_dim': 32,
        'n_timesteps': 600,
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
    # transformList = []
    # transformList.append(transforms.ToTensor())
    # basic_transforms = transforms.Compose(transformList)
    basic_transforms = T.Compose([
        T.ToTensor(),       # Converts [0, 255] to [0.0, 1.0]
        T.Lambda(lambda t: (t * 2) - 1)
    ])

    datasetRoot = "DIV2K"
    lr_path = datasetRoot+"/DIV2K_train_LR_x8"
    hr_path = datasetRoot+"/DIV2K_train_HR"

    train_dataset = GANDIV2KDataLoader(
        root_dir_lr=lr_path,
        root_dir_hr=hr_path,
        transformLr=basic_transforms,
        transformHr=basic_transforms,
        mode="train",
        batch_size=16,
        scale=8,
        patch_size=80,
    )

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

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=2,
        pin_memory=True,
        num_workers=cpu_count(),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=cpu_count(),
        pin_memory=True,
    )

    # Define the Binary Cross Entropy loss function
    loss_func = nn.MSELoss()

    model = SR3UNet(in_channels = 3,
                     cond_channels = 3,
                     base_channels = config['hidden_dim'],
                     time_dim = config['timestep_embedding_dim']
                     ).to(device)
    diffusion = Diffusion(model,  
                          timesteps=config['n_timesteps'],
                          device=device
                          )
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    trainer = Trainer(loss_func, model, diffusion, train_loader, val_loader, optimizer, device)
    trainer.loopEpochs(config['epochs'])

    alert.send_notification(f"Training finished for visual AI")

def debug_step_stats(x0, cond, t, noise, predicted_noise):
    # x0, predicted_noise, noise: tensors
    with torch.no_grad():
        print("x0 range:", x0.min().item(), x0.max().item())
        print("cond range:", cond.min().item(), cond.max().item())
        print("noise mean/std:", noise.mean().item(), noise.std().item())
        print("pred_noise mean/std:", predicted_noise.mean().item(), predicted_noise.std().item())
        # correlation between predicted_noise and noise
        pn = predicted_noise.view(predicted_noise.size(0), -1)
        n = noise.view(noise.size(0), -1)
        corr = (pn * n).mean() / (pn.std(dim=1).mean() * n.std(dim=1).mean() + 1e-8)
        print("approx cross-corr:", corr.item())
        
def reconstruct_x0_from_xt(x_t, predicted_noise, t, alpha_cumprod):
    # x_t, predicted_noise: (B,C,H,W)
    ac = alpha_cumprod[t].view(-1,1,1,1)
    sqrt_ac = ac.sqrt()
    sqrt_1_ac = (1.0 - ac).sqrt()
    x0_pred = (x_t - sqrt_1_ac * predicted_noise) / (sqrt_ac + 1e-8)
    return x0_pred

def psnr_torch(x_pred, x_true, data_range=1.0):
    mse = F.mse_loss(x_pred, x_true, reduction='mean')
    if mse == 0:
        return float('inf')
    return (20 * torch.log10(torch.tensor(data_range)) - 10 * torch.log10(mse)).item()


class Trainer:
    def __init__(self, loss_func, model, diffusion, train_loader, val_loader, optimizer, device):
        self.model = model        
        self.diffusion = diffusion
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        # self.scheduler = scheduler
        self.device = device
        self.best_psnr = 0.0
        self.scale = 8

        # Directories
        self.checkpoint_dir = 'super-resolution-model-tests/diffusion/training_checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.image_dir = 'super-resolution-model-tests/diffusion/generated_image/'
        os.makedirs(self.image_dir, exist_ok=True)

    def compute_mse(self, img1, img2):
        return F.mse_loss(img1, img2)

    def compute_psnr(self, mse):
        mse_tensor = torch.tensor(mse) 
        return 10 * torch.log10(1 / mse_tensor)

    def loopEpochs(self, num_epoch):
        self.model.train()
        scaler = torch.amp.GradScaler('cuda')
        # accumulation_steps = 16 # Virtual Batch Size = 8 * 4 = 32

        for epoch in range(num_epoch):
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            start_time = time.time()
            total_loss = 0

            for idx, (lr_img, hr_img) in enumerate(self.train_loader):
                lr_img = lr_img.to(self.device)
                hr_img = hr_img.to(self.device)

                lr_up = F.interpolate(lr_img, size=hr_img.shape[-2:], mode="bicubic")

                t = torch.randint(0, self.diffusion.timesteps, (hr_img.size(0),), device=self.device).long()
                x_t, noise = self.diffusion.q_sample(hr_img, t)

                self.optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    pred_noise = self.model(x_t, lr_up, t)
                    loss = self.loss_func(pred_noise, noise)

                scaler.scale(loss).backward()
                # if (idx + 1) % accumulation_steps == 0:
                #     scaler.step(self.optimizer)
                #     scaler.update()
                #     self.optimizer.zero_grad()
                #     torch.cuda.empty_cache()
                
                # # Multiply back for logging
                # total_loss += loss.item() * accumulation_steps
                scaler.step(self.optimizer)
                scaler.update()
                torch.cuda.empty_cache()
                total_loss += loss

            # losses once per epoch
            print(f'Epoch [{epoch + 1}/{num_epoch}]  | Loss {total_loss / (len(self.train_loader))} | Time: {time.time() - start_time:.2f} sec')
            if (epoch + 1) % 10 == 0:
                # debug_step_stats(hr_img, lr_img, t, noise, pred_noise)
                x0_pred = reconstruct_x0_from_xt(x_t, pred_noise, t, self.diffusion.alpha_cumprod.to(hr_img.device))

                # If your data is in [0,1], just clamp
                x0_pred = x0_pred.clamp(0.0, 1.0)
                x0_gt = hr_img.clamp(0.0, 1.0)

                print("PSNR (reconstruction at t=10):", psnr_torch(x0_pred, x0_gt, data_range=1.0))

            # Visualise the generated image at different epochs
            if ((epoch + 1) % 50 == 0 and epoch >= 500) or (epoch + 1 == 50):
                # self.visualise_generated_images(epoch)
                self.visualise_validation_set(epoch)
            
            # if self.scheduler is not None:
            #     self.scheduler.step()

    def visualise_validation_set(self, epoch):
        self.model.eval()
        total_psnr = 0.0
        count = 0
        
        # Settings
        scale_factor = self.val_loader.dataset.scale
        # LR tile size. Output will be tile_size * scale (e.g. 32*8 = 256)
        # 32 is a safe number for VRAM. You can try 64 if you have a 3090/4090.
        lr_tile_size = 32 
        
        # Only process a few images to save time, or remove this to process all
        num_images_to_check = 1 
        imageChosenNum = random.randint(0, num_images_to_check - 1)

        for idx, (lr_img_full, hr_img_full) in enumerate(self.val_loader):
            if idx >= num_images_to_check:
                break
                
            start_time = time.time()
            lr_img_full = lr_img_full.to(self.device)
            hr_img_full = hr_img_full.to(self.device)
            
            # 1. PREPARE TILES
            B, C, H, W = lr_img_full.shape
            
            # Calculate how many tiles we need
            h_steps = (H // lr_tile_size) + (1 if H % lr_tile_size != 0 else 0)
            w_steps = (W // lr_tile_size) + (1 if W % lr_tile_size != 0 else 0)
            
            patches = []
            coords = [] # Keep track of where to put them back

            for i in range(h_steps):
                for j in range(w_steps):
                    # Define crop coordinates
                    h_start = i * lr_tile_size
                    h_end = min(h_start + lr_tile_size, H)
                    w_start = j * lr_tile_size
                    w_end = min(w_start + lr_tile_size, W)
                    
                    # Crop LR patch
                    patch = lr_img_full[:, :, h_start:h_end, w_start:w_end]
                    
                    # Pad if the patch is smaller than tile size (edge cases)
                    pad_h = lr_tile_size - (h_end - h_start)
                    pad_w = lr_tile_size - (w_end - w_start)
                    if pad_h > 0 or pad_w > 0:
                        patch = F.pad(patch, (0, pad_w, 0, pad_h), mode='constant', value=0)
                        
                    patches.append(patch)
                    coords.append((h_start, h_end, w_start, w_end, pad_h, pad_w))

            # 2. BATCH INFERENCE (The Speedup)
            # Stack all patches into one tensor: (Num_Tiles, 3, 32, 32)
            batch_patches = torch.cat(patches, dim=0)
            
            with torch.no_grad():
                # Run Diffusion ONCE on the whole batch
                # Output shape: (Num_Tiles, 3, 256, 256)
                sr_patches = self.diffusion.sample(
                    batch_patches, 
                    shape=(batch_patches.shape[0], 3, lr_tile_size*scale_factor, lr_tile_size*scale_factor)
                )

            # 3. STITCH BACK TOGETHER
            # Create empty canvas
            H_sr, W_sr = H * scale_factor, W * scale_factor
            sr_img_full = torch.zeros((1, 3, H_sr, W_sr), device=self.device)

            for k in range(len(coords)):
                h_start, h_end, w_start, w_end, pad_h, pad_w = coords[k]
                
                # Get the generated patch
                sr_patch = sr_patches[k]
                
                # Remove padding if we added it (output side)
                if pad_h > 0 or pad_w > 0:
                    sr_patch = sr_patch[:, :sr_patch.shape[1] - (pad_h*scale_factor), :sr_patch.shape[2] - (pad_w*scale_factor)]

                # Place into canvas
                sr_img_full[:, :, h_start*scale_factor:h_end*scale_factor, w_start*scale_factor:w_end*scale_factor] = sr_patch

            # 4. METRICS & VISUALIZATION (Standard logic)
            hr_img_vis = (hr_img_full.squeeze(0).clamp(-1, 1) + 1) / 2
            sr_img_vis = (sr_img_full.squeeze(0).clamp(-1, 1) + 1) / 2
            
            # Align
            H = min(hr_img_vis.shape[1], sr_img_vis.shape[1])
            W = min(hr_img_vis.shape[2], sr_img_vis.shape[2])
            hr_img_vis = hr_img_vis[:, :H, :W]
            sr_img_vis = sr_img_vis[:, :H, :W]

            # Compute PSNR
            mse_sr = self.compute_mse(hr_img_vis, sr_img_vis).item()
            psnr_sr = self.compute_psnr(mse_sr).item()
            total_psnr += psnr_sr
            count += 1
            
            print(f'Image Build Time: {time.time() - start_time:.2f} sec')

            # Save Grid (Only for the chosen image)
            if idx == imageChosenNum:
                lr_resized_vis = (F.interpolate(lr_img_full, scale_factor=scale_factor, mode='nearest').squeeze(0).clamp(-1, 1) + 1) / 2
                comparison = torch.stack([lr_resized_vis, sr_img_vis, hr_img_vis], dim=0)
                grid = torchvision.utils.make_grid(comparison, nrow=3, value_range=(0,1))
                Image.fromarray((grid.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)).save(f"{self.image_dir}/top_image.png")

        avg_psnr = total_psnr / count if count > 0 else 0.0
        print(f'Validation Average PSNR (Full Image Tiling): {avg_psnr:.4f}')

        # Save Checkpoint Logic...
        if self.best_psnr < avg_psnr:
             self.best_psnr = avg_psnr
             save_checkpoint(epoch, avg_psnr, self.model, self.checkpoint_dir)
             alert.send_notification(f"New best AVG PSNR: Epoch {epoch + 1}  PSNR: {avg_psnr:.2f}")
        
        self.model.train()
    # def visualise_validation_set(self, epoch):
    #     self.model.eval()

    #     total_psnr = 0.0
    #     count = 0
    #     imageChosen = None
    #     scale_factor = self.val_loader.dataset.scale
    #     num_images = 1
    #     # num_images = self.val_loader.dataset.__len__() // 10 # only check on a 5th of the set
    #     imageChosenNum = random.randint(0, num_images - 1)

    #     # Loop over the entire validation loader
    #     for idx, (lr_img_full, hr_img_full) in enumerate(self.val_loader):
    #         start_time = time.time()
    #         lr_img_full = lr_img_full.to(self.device)
    #         hr_img_full = hr_img_full.to(self.device)
            
    #         with torch.no_grad():
    #             B, C, H, W = lr_img_full.shape
    #             sr_img = self.diffusion.sample(lr_img_full, shape=(B, C, H*self.scale, W*self.scale)).to(self.device).squeeze(0)
                
    #         hr_img_vis = hr_img_full.squeeze(0)
    #         hr_img_vis = (hr_img_vis.clamp(-1, 1) + 1) / 2
    #         # sr_img_vis = sr_img.clamp(0, 1)
    #         sr_img_vis = (sr_img.clamp(-1, 1) + 1) / 2

    #         # Align sizing issues from patching
    #         H = min(hr_img_vis.shape[1], sr_img_vis.shape[1])
    #         W = min(hr_img_vis.shape[2], sr_img_vis.shape[2])
    #         hr_img_vis = hr_img_vis[:, :H, :W]
    #         sr_img_vis = sr_img_vis[:, :H, :W]

    #         # Calculate PSNR for this image
    #         mse_sr = self.compute_mse(hr_img_vis, sr_img_vis).item()
    #         psnr_sr = self.compute_psnr(mse_sr).item()
        
    #         # Accumulate PSNR
    #         total_psnr += psnr_sr
    #         count += 1

    #         print(f'Image Build Time: {time.time() - start_time:.2f} sec')

    #         # Grid
    #         if idx == imageChosenNum:
    #             lr_resized_vis = (F.interpolate(lr_img_full, 
    #                                            scale_factor=scale_factor, 
    #                                            mode='nearest').squeeze(0).clamp(-1, 1) + 1) / 2

    #             comparison = torch.stack([lr_resized_vis, sr_img_vis, hr_img_vis], dim=0)
    #             grid = torchvision.utils.make_grid(comparison, nrow=3, value_range=(0,1))
    #             grid_np = grid.permute(1,2,0).cpu().numpy()
    #             first_image_grid = (grid_np * 255).astype(np.uint8)

    #         if num_images - 1 == idx:
    #             break

    #     avg_psnr = total_psnr / count if count > 0 else 0.0

    #     # Save Checkpoint
    #     if self.best_psnr < avg_psnr:
    #         self.best_psnr = avg_psnr
    #         save_checkpoint(epoch, avg_psnr, self.model, self.checkpoint_dir)
    #         alert.send_notification(f"New best AVG PSNR: Epoch {epoch + 1}  PSNR: {avg_psnr:.2f}")

    #     filename = f"{self.image_dir}/top_image.png"
    #     Image.fromarray(firVkst_image_grid).save(filename)

    #     # Print Final PSNR value
    #     print(f'Validation Average PSNR (Full Image Tiling): {avg_psnr:.4f}')

    #     self.model.train()


    # def visualise_validation_set(self, epoch):
    #     self.model.eval()

    #     total_psnr = 0.0
    #     count = 0
    #     imageChosen = None
    #     num_images = self.val_loader.dataset.__len__() // 20 # only check on a 5th of the set
    #     imageChosenNum = random.randint(0, num_images - 1)

    #     # Loop over the entire validation loader
    #     for idx, (lr_img_full, hr_img_full) in enumerate(self.val_loader):
    #         # lr_img_full = lr_img_full.to(self.device)
    #         # hr_img_full = hr_img_full.to(self.device)
    #         _, _, H_lr, W_lr = lr_img_full.shape
        
    #         TILE_SIZE = 128

    #         # Calculate number of tiles needed
    #         num_h = H_lr // TILE_SIZE + (1 if H_lr % TILE_SIZE != 0 else 0)
    #         num_w = W_lr // TILE_SIZE + (1 if W_lr % TILE_SIZE != 0 else 0)

    #         # Create an empty tensor to store the stitched SR image
    #         H_sr = H_lr * self.scale
    #         W_sr = W_lr * self.scale
    #         sr_img_stitched = torch.zeros(hr_img_full.shape[1], H_sr, W_sr, dtype=hr_img_full.dtype)
            
    #         with torch.no_grad():
    #             for i in range(num_h):
    #                 for j in range(num_w):
    #                     # Patch
    #                     start_h_lr = i * TILE_SIZE
    #                     end_h_lr = min((i + 1) * TILE_SIZE, H_lr)
    #                     start_w_lr = j * TILE_SIZE
    #                     end_w_lr = min((j + 1) * TILE_SIZE, W_lr)
                    
    #                     lr_patch = lr_img_full[:, :, start_h_lr:end_h_lr, start_w_lr:end_w_lr].to(self.device)
                    
    #                     # Generate
    #                     B, C, H, W = lr_patch.shape
    #                     sr_patch = self.diffusion.sample(lr_patch, shape=(B, C, H*self.scale, W*self.scale)).to(self.device).squeeze(0)

    #                     # Stitch
    #                     start_h_sr = start_h_lr * self.scale
    #                     end_h_sr = end_h_lr * self.scale
    #                     start_w_sr = start_w_lr * self.scale
    #                     end_w_sr = end_w_lr * self.scale

    #                     sr_img_stitched[:, start_h_sr:end_h_sr, start_w_sr:end_w_sr] = sr_patch

    #         hr_img_vis = hr_img_full.squeeze(0).clamp(0, 1)
    #         sr_img_vis = sr_img_stitched.clamp(0, 1)

    #         # Align sizing issues from patching
    #         H = min(hr_img_vis.shape[1], sr_img_vis.shape[1])
    #         W = min(hr_img_vis.shape[2], sr_img_vis.shape[2])
    #         hr_img_vis = hr_img_vis[:, :H, :W]
    #         sr_img_vis = sr_img_vis[:, :H, :W]

    #         # Calculate PSNR for this image
    #         mse_sr = self.compute_mse(hr_img_vis, sr_img_vis).item()
    #         psnr_sr = self.compute_psnr(mse_sr).item()
        
    #         # Accumulate PSNR
    #         total_psnr += psnr_sr
    #         count += 1

    #         # Grid
    #         if idx == imageChosenNum:
    #             lr_resized_vis = F.interpolate(lr_img_full, 
    #                                             scale_factor=self.scale, 
    #                                             mode='nearest').squeeze(0).clamp(0, 1)

    #             comparison = torch.stack([lr_resized_vis, sr_img_vis, hr_img_vis], dim=0)
    #             grid = torchvision.utils.make_grid(comparison, nrow=3, value_range=(0,1))
    #             grid_np = grid.permute(1,2,0).numpy()
    #             first_image_grid = (grid_np * 255).astype(np.uint8)
            
    #         if idx == num_images:
    #             break

    #     avg_psnr = total_psnr / count if count > 0 else 0.0

    #     # Save Checkpoint
    #     if self.best_psnr < avg_psnr:
    #         self.best_psnr = avg_psnr
    #         save_checkpoint(epoch, avg_psnr, self.model, self.checkpoint_dir)
    #         alert.send_notification(f"New best AVG PSNR: Epoch {epoch + 1}  PSNR: {avg_psnr:.2f}")

    #     filename = f"{self.image_dir}/top_image.png"
    #     Image.fromarray(first_image_grid).save(filename)

    #     # Print Final PSNR value
    #     print(f'Validation Average PSNR (Full Image Tiling): {avg_psnr:.4f}')

    #     self.model.train()

if __name__ == "__main__":
    main()
