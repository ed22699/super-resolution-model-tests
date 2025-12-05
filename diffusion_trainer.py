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
from dataloader import GANDIV2KDataLoader
from diffusion import Diffusion,  SR3UNet
from utils.save_checkpoint_diff import save_checkpoint
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random
import argparse
from torchvision.transforms import v2

# Alert messages
# import alert

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
parser.add_argument(
    "--scale",
    default=8,
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
        scale=args.scale,
        patch_size=80,
    )
        

    lr_path = datasetRoot+"/DIV2K_valid_LR_x8"
    hr_path = datasetRoot+"/DIV2K_valid_HR"

    val_dataset = GANDIV2KDataLoader(
        root_dir_lr=lr_path,
        root_dir_hr=hr_path,
        transformLr=basic_transforms_Hr,
        transformHr=basic_transforms_Hr,
        mode="val",
        batch_size=4,
        scale=args.scale,
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

    # Setup Model
    loss_func = nn.MSELoss()

    model = SR3UNet(in_channels = 3,
                     cond_channels = 3,
                     base_channels = args.hidden_dim,
                     channel_mults=[1,2,4,8,16],
                     time_dim = args.timestep_embedding_dim
                     ).to(device)

    # Code to load in a checkpoint and continue training
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

    # alert.send_notification(f"Training finished for visual AI")

class Trainer:
    def __init__(self, loss_func, model, diffusion, train_loader, val_loader, optimizer, device, startEpoch=0):
        self.model = model        
        self.diffusion = diffusion
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.best_psnr = 0.0
        self.scale = 8
        self.startEpoch = startEpoch

        # Directories
        self.checkpoint_dir = 'super-resolution-model-tests/diffusion/training_checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.image_dir = 'super-resolution-model-tests/diffusion/generated_image/'
        os.makedirs(self.image_dir, exist_ok=True)

    def compute_psnr(self, img1, img2, max_val=1.0):
        mse = F.mse_loss(img1, img2)
        psnr = 10 * torch.log10((max_val ** 2) / mse)
        return psnr

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

                # Decide timestep
                t = torch.randint(0, self.diffusion.timesteps, (hr_img.size(0),), device=self.device).long()
                x_t, noise = self.diffusion.q_sample(hr_img, t)

                self.optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    pred_noise = self.model(x_t, lr_up, t)
                    loss = self.loss_func(pred_noise, noise)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                torch.cuda.empty_cache()
                total_loss += loss

            # losses once per epoch
            print(f'Epoch [{epoch + 1 + self.startEpoch}/{num_epoch}]  | Loss {total_loss / (len(self.train_loader))} | Time: {time.time() - start_time:.2f} sec')

            # Visualise the generated image at different epochs
            if ((epoch + 1 + self.startEpoch) % 50 == 0 and epoch + self.startEpoch >= 500) or (epoch + 1 + self.startEpoch == 50):
                self.visualise_validation_set(epoch + self.startEpoch)
            

    def visualise_validation_set(self, epoch):
        self.model.eval()
        total_psnr = 0.0
        count = 0
        
        # Get scale from dataset
        scale_factor = self.val_loader.dataset.scale 

        lr_tile_size = 32 
        
        num_images_to_check = 1 # Done due to large creation time
        imageChosenNum = random.randint(0, num_images_to_check - 1)

        for idx, (lr_img_full, hr_img_full) in enumerate(self.val_loader):
            if idx >= num_images_to_check:
                break
            
            start_time = time.time()
            
            # Calculate steps
            B, C, H, W = lr_img_full.shape
            
            h_steps = (H // lr_tile_size) + (1 if H % lr_tile_size != 0 else 0)
            w_steps = (W // lr_tile_size) + (1 if W % lr_tile_size != 0 else 0)
            
            patches = []
            coords = [] 

            for i in range(h_steps):
                for j in range(w_steps):
                    h_start = i * lr_tile_size
                    h_end = min(h_start + lr_tile_size, H)
                    w_start = j * lr_tile_size
                    w_end = min(w_start + lr_tile_size, W)
                    
                    # Crop LR
                    patch = lr_img_full[:, :, h_start:h_end, w_start:w_end]
                    
                    # Pad if edge tile smaller than tile size
                    pad_h = lr_tile_size - (h_end - h_start)
                    pad_w = lr_tile_size - (w_end - w_start)
                    if pad_h > 0 or pad_w > 0:
                        patch = F.pad(patch, (0, pad_w, 0, pad_h), mode='constant', value=0)
                        
                    patches.append(patch)
                    coords.append((h_start, h_end, w_start, w_end, pad_h, pad_w))

            # Stack patches
            batch_patches = torch.cat(patches, dim=0).to(self.device)
            
            sr_patches_list = []
            mini_batch = 4 
            
            with torch.no_grad():
                for k in range(0, len(batch_patches), mini_batch):
                    batch_slice = batch_patches[k : k + mini_batch]
                    
                    # Diffusion Sampling
                    output_slice = self.diffusion.sample(
                        batch_slice, 
                        shape=(batch_slice.shape[0], 3, lr_tile_size*scale_factor, lr_tile_size*scale_factor)
                    )
                    sr_patches_list.append(output_slice.cpu())

            sr_patches_all = torch.cat(sr_patches_list, dim=0)

            # Stich patches
            H_sr, W_sr = H * scale_factor, W * scale_factor
            sr_img_full = torch.zeros((1, 3, H_sr, W_sr), device='cpu')

            for k in range(len(coords)):
                h_start, h_end, w_start, w_end, pad_h, pad_w = coords[k]
                sr_patch = sr_patches_all[k]
                
                # Remove padding
                if pad_h > 0 or pad_w > 0:
                    sr_patch = sr_patch[:, :sr_patch.shape[1] - (pad_h*scale_factor), :sr_patch.shape[2] - (pad_w*scale_factor)]

                sr_img_full[:, :, h_start*scale_factor:h_end*scale_factor, w_start*scale_factor:w_end*scale_factor] = sr_patch

            hr_img_vis = (hr_img_full.cpu().squeeze(0).clamp(-1, 1) + 1) / 2
            sr_img_vis = (sr_img_full.squeeze(0).clamp(-1, 1) + 1) / 2
            
            # Align dimensions (HR might be slightly larger due to tile padding mismatch on LR)
            H_vis = min(hr_img_vis.shape[1], sr_img_vis.shape[1])
            W_vis = min(hr_img_vis.shape[2], sr_img_vis.shape[2])
            hr_img_vis = hr_img_vis[:, :H_vis, :W_vis]
            sr_img_vis = sr_img_vis[:, :H_vis, :W_vis]

            # Compute PSNR
            psnr_sr = self.compute_psnr(hr_img_vis, sr_img_vis).item()
            total_psnr += psnr_sr
            count += 1
            
            print(f'Validation Image Time: {time.time() - start_time:.2f} sec')

            # Save Visualization Grid (Only if this is the chosen image)
            if idx == imageChosenNum:
                # Resize LR for comparison
                lr_resized_vis = (F.interpolate(lr_img_full.cpu(), scale_factor=scale_factor, mode='nearest').squeeze(0).clamp(-1, 1) + 1) / 2
                lr_resized_vis = lr_resized_vis[:, :H_vis, :W_vis]

                comparison = torch.stack([lr_resized_vis, sr_img_vis, hr_img_vis], dim=0)
                grid = torchvision.utils.make_grid(comparison, nrow=3, value_range=(0,1))
                
                # Save
                Image.fromarray((grid.permute(1,2,0).numpy() * 255).astype(np.uint8)).save(f"{self.image_dir}/top_image.png")

        avg_psnr = total_psnr / count if count > 0 else 0.0
        print(f'Validation Average PSNR: {avg_psnr:.4f}')

        # Save Best Model Logic
        if self.best_psnr < avg_psnr:
             self.best_psnr = avg_psnr
             save_checkpoint(epoch, avg_psnr, self.model, self.checkpoint_dir)
            #  alert.send_notification(f"New best AVG PSNR: Epoch {epoch + 1}  PSNR: {avg_psnr:.2f}")
        
        self.model.train()

if __name__ == "__main__":
    main(parser.parse_args())