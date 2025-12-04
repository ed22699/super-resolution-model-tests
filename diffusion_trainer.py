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
import argparse
from torchvision.transforms import v2

# Alert messages
import alert

parser = argparse.ArgumentParser(
    description="Train Diffusion model",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--timestep-embedding-dim",
    default=256,
    type=int,
)
parser.add_argument(
    "--hidden-dim",
    default=64,
    type=int,
)
parser.add_argument(
    "--n-timesteps",
    default=1000,
    type=int,
)
parser.add_argument(
    "--lr",
    default=1e-4,
    type=float,
)
parser.add_argument(
    "--epochs",
    default=1000,
    type=int,
)

def main(args):
    # For GPU (CUDA) or CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess the dataset
    basic_transforms_Hr = T.Compose([
        T.ToTensor(),      
        T.Lambda(lambda t: (t * 2) - 1)
    ])
    basic_transforms_Lr = T.Compose([
        T.ToTensor(),  
        v2.GaussianNoise(mean=0.0, sigma=0.3, clip=True),
        T.Lambda(lambda t: (t * 2) - 1)
    ])

    datasetRoot = "DIV2K"
    lr_path = datasetRoot+"/DIV2K_train_LR_x8"
    hr_path = datasetRoot+"/DIV2K_train_HR"

    train_dataset = GANDIV2KDataLoader(
        root_dir_lr=lr_path,
        root_dir_hr=hr_path,
        transformLr=basic_transforms_Hr,
        transformHr=basic_transforms_Hr,
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
        transformLr=basic_transforms_Lr,
        transformHr=basic_transforms_Hr,
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
                     base_channels = args.hidden_dim,
                     time_dim = args.timestep_embedding_dim
                     ).to(device)

    # checkpoint = "ckpt_PSNR_19.1473.pth"
    # checkpointPath = "super-resolution-model-tests/diffusion/training_checkpoints/"+checkpoint
    # state_dict = torch.load(checkpointPath, map_location=device)
    # model.load_state_dict(state_dict['model_state_dict'])
    # startEpoch = state_dict['epoch']

    diffusion = Diffusion(model,  
                          timesteps=args.n_timesteps,
                          device=device
                          )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    trainer = Trainer(loss_func, model, diffusion, train_loader, val_loader, optimizer, device)
    trainer.loopEpochs(args.epochs)

    alert.send_notification(f"Training finished for visual AI")

class Trainer:
    def __init__(self, loss_func, model, diffusion, train_loader, val_loader, optimizer, device, startEpoch=0):
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
        self.startEpoch = startEpoch

        # Directories
        self.checkpoint_dir = 'super-resolution-model-tests/diffusion/n3/training_checkpoints'
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

        for epoch in range(num_epoch - self.startEpoch):
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
            print(f'Epoch [{epoch + 1 + self.startEpoch}/{num_epoch}]  | Loss {total_loss / (len(self.train_loader))} | Time: {time.time() - start_time:.2f} sec')

            # Visualise the generated image at different epochs
            if ((epoch + 1 + self.startEpoch) % 50 == 0 and epoch + self.startEpoch >= 500) or (epoch + 1 + self.startEpoch == 50):
                # self.visualise_generated_images(epoch)
                self.visualise_validation_set(epoch + self.startEpoch)
            
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
        lr_tile_size = 64 
        
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
    #     total_psnr = 0.0
    #     count = 0
    #     imageChosen = None
    #     scale_factor = self.val_loader.dataset.scale

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

    #         # Save image
    #         lr_resized_vis = (F.interpolate(lr_img_full, scale_factor=scale_factor, mode='nearest').squeeze(0).clamp(-1, 1) + 1) / 2
    #         comparison = torch.stack([lr_resized_vis, sr_img_vis, hr_img_vis], dim=0)
    #         grid = torchvision.utils.make_grid(comparison, nrow=3, value_range=(0,1))
    #         Image.fromarray((grid.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)).save(f"{self.image_dir}/top_image.png")
    #         break

    #     avg_psnr = total_psnr / count if count > 0 else 0.0

    #     # Print Final PSNR value
    #     print(f"Diffusion test complete: Avg PSNR {avg_psnr:.2f}")


    # def visualise_validation_set(self, epoch):
    #     self.model.eval()
    #     total_psnr = 0.0
    #     count = 0
        
    #     scale_factor = self.val_loader.dataset.scale
    #     # Increase tile size if VRAM allows (64 is better than 32)
    #     lr_tile_size = 32 
    #     # Context buffer: how many pixels to look around the tile
    #     buffer = 8 
    #     # ----------------------------------------

    #     num_images_to_check = 1 
    #     imageChosenNum = random.randint(0, num_images_to_check - 1)

    #     for idx, (lr_img_full, hr_img_full) in enumerate(self.val_loader):
    #         if idx >= num_images_to_check:
    #             break
                
    #         start_time = time.time()
    #         lr_img_full = lr_img_full.to(self.device)
    #         hr_img_full = hr_img_full.to(self.device)
            
    #         B, C, H, W = lr_img_full.shape
            
    #         # Calculate grid
    #         h_steps = (H // lr_tile_size) + (1 if H % lr_tile_size != 0 else 0)
    #         w_steps = (W // lr_tile_size) + (1 if W % lr_tile_size != 0 else 0)
            
    #         patches = []
    #         coords = [] 

    #         for i in range(h_steps):
    #             for j in range(w_steps):
    #                 # 1. Determine the "Core" tile (the part we want to keep)
    #                 h_start = i * lr_tile_size
    #                 h_end = min(h_start + lr_tile_size, H)
    #                 w_start = j * lr_tile_size
    #                 w_end = min(w_start + lr_tile_size, W)
                    
    #                 # 2. Determine the "Buffered" tile (Core + Neighbors)
    #                 # We expand the crop by 'buffer' amount, being careful of image edges
    #                 h_start_buf = max(h_start - buffer, 0)
    #                 h_end_buf = min(h_end + buffer, H)
    #                 w_start_buf = max(w_start - buffer, 0)
    #                 w_end_buf = min(w_end + buffer, W)
                    
    #                 # Crop the buffered patch
    #                 patch = lr_img_full[:, :, h_start_buf:h_end_buf, w_start_buf:w_end_buf]
                    
    #                 # 3. Save info to un-crop later
    #                 # We need to know how much padding we actually got on each side
    #                 pad_top = h_start - h_start_buf
    #                 pad_bottom = h_end_buf - h_end
    #                 pad_left = w_start - w_start_buf
    #                 pad_right = w_end_buf - w_end

    #                 patches.append(patch)
    #                 coords.append((h_start, h_end, w_start, w_end, pad_top, pad_bottom, pad_left, pad_right))

    #         # 4. BATCH INFERENCE
    #         # Note: Patches are different sizes now (edges vs center), so we can't strict stack.
    #         # We process them in a loop or pad them to max size. 
    #         # Ideally, we pad them all to be the same size for batching.
            
    #         # Simple Batching Strategy: Pad everyone to max_size
    #         max_h = lr_tile_size + 2*buffer
    #         max_w = lr_tile_size + 2*buffer
            
    #         batch_patches = []
    #         for p in patches:
    #             # Pad patch to max size (bottom-right padding)
    #             ph = max_h - p.shape[2]
    #             pw = max_w - p.shape[3]
    #             p_padded = F.pad(p, (0, pw, 0, ph), mode='reflect') # Reflect is safe here inside the image context
    #             batch_patches.append(p_padded)
            
    #         batch_tensor = torch.cat(batch_patches, dim=0)

    #         with torch.no_grad():
    #             # Run Diffusion on Buffered Tiles
    #             sr_patches_padded = self.diffusion.sample(
    #                 batch_tensor, 
    #                 shape=(batch_tensor.shape[0], 3, max_h*scale_factor, max_w*scale_factor)
    #             )

    #         # 5. STITCH AND CROP
    #         H_sr, W_sr = H * scale_factor, W * scale_factor
    #         sr_img_full = torch.zeros((1, 3, H_sr, W_sr), device=self.device)

    #         for k in range(len(coords)):
    #             h_start, h_end, w_start, w_end, pt, pb, pl, pr = coords[k]
                
    #             # The raw output from model
    #             sr_patch_raw = sr_patches_padded[k]
                
    #             # 1. Remove the "Batching Padding" (bottom/right) we added in step 4
    #             valid_h = (h_end - h_start) + pt + pb
    #             valid_w = (w_end - w_start) + pl + pr
    #             sr_patch_valid = sr_patch_raw[:, :valid_h*scale_factor, :valid_w*scale_factor]

    #             # 2. Remove the "Context Buffer" (the edges) to get the clean center
    #             # Crop: Top, Bottom, Left, Right
    #             crop_top = pt * scale_factor
    #             crop_bottom = sr_patch_valid.shape[1] - (pb * scale_factor)
    #             crop_left = pl * scale_factor
    #             crop_right = sr_patch_valid.shape[2] - (pr * scale_factor)
                
    #             final_patch = sr_patch_valid[:, crop_top:crop_bottom, crop_left:crop_right]

    #             # Place into canvas
    #             sr_img_full[:, :, h_start*scale_factor:h_end*scale_factor, w_start*scale_factor:w_end*scale_factor] = final_patch

    #         # 6. VISUALIZE
    #         hr_img_vis = (hr_img_full.squeeze(0).clamp(-1, 1) + 1) / 2
    #         sr_img_vis = (sr_img_full.squeeze(0).clamp(-1, 1) + 1) / 2

    #         mse_sr = self.compute_mse(hr_img_vis, sr_img_vis).item()
    #         psnr_sr = self.compute_psnr(mse_sr).item()
    #         total_psnr += psnr_sr
    #         count += 1
            
    #         print(f'Image Build Time: {time.time() - start_time:.2f} sec')

    #         if idx == imageChosenNum:
    #             lr_resized_vis = (F.interpolate(lr_img_full, scale_factor=scale_factor, mode='nearest').squeeze(0).clamp(-1, 1) + 1) / 2
    #             comparison = torch.stack([lr_resized_vis, sr_img_vis, hr_img_vis], dim=0)
    #             grid = torchvision.utils.make_grid(comparison, nrow=3, value_range=(0,1))
    #             Image.fromarray((grid.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)).save(f"{self.image_dir}/epoch_{epoch}.png")

    #     avg_psnr = total_psnr / count if count > 0 else 0.0
    #     print(f'Validation Average PSNR (Full Image Tiling): {avg_psnr:.4f}')
        
    #     if self.best_psnr < avg_psnr:
    #          self.best_psnr = avg_psnr
    #          save_checkpoint(epoch, avg_psnr, self.model, self.checkpoint_dir)
    #          alert.send_notification(f"New best AVG PSNR: Epoch {epoch + 1}  PSNR: {avg_psnr:.2f}")
        
    #     self.model.train()

if __name__ == "__main__":
    main(parser.parse_args())