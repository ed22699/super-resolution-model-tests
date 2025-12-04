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
from transformer import SwinIR
from utils.save_checkpoint_diff import save_checkpoint
from gan_dataloader import GANDIV2KDataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random
import argparse
from torchvision.transforms import v2

# Alert messages
import alert


parser = argparse.ArgumentParser(
    description="Train Transformer model",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--embed-dim",
    default=64,
    type=int,
)
parser.add_argument(
    "--upscale",
    default=8,
    type=int,
)
parser.add_argument(
    "--lr",
    default=1e-4,
    type=float,
)
parser.add_argument(
    "--epochs",
    default=5000,
    type=int,
)

def main(args):

    # For GPU (CUDA) or CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess the dataset
    basic_transforms_Lr = T.Compose([
        T.ToTensor(),  
        v2.GaussianNoise(mean=0.0, sigma=0.3, clip=True)
    ])
    basic_transforms_Hr = T.Compose([
        T.ToTensor(),       # Converts [0, 255] to [0.0, 1.0]
    ])

    datasetRoot = "DIV2K"
    lr_path = datasetRoot+"/DIV2K_train_LR_x8"
    hr_path = datasetRoot+"/DIV2K_train_HR"

    train_dataset = GANDIV2KDataLoader(
        root_dir_lr=lr_path,
        root_dir_hr=hr_path,
        transformLr=basic_transforms_Lr,
        transformHr=basic_transforms_Hr,
        mode="train",
        batch_size=16,
        scale=8,
        patch_size=64,
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
        batch_size=4,
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


    model = SwinIR(in_ch = 3,
                     embed_dim=args.embed_dim,
                     upscale = args.upscale
                     ).to(device)

    # Define Loss and scheduler
    loss_func = nn.L1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    trainer = Trainer(loss_func, model, scheduler, train_loader, val_loader, optimizer, device, args.epochs)
    trainer.loopEpochs(args.epochs)

    alert.send_notification(f"Training finished for Transformer")

class Trainer:
    def __init__(self, loss_func, model, scheduler, train_loader, val_loader, optimizer, device, epochs):
        self.model = model        
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.best_psnr = 0.0
        self.scale = 8
        self.epochs = epochs

        # Directories
        self.checkpoint_dir = 'super-resolution-model-tests/transformer/training_checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.image_dir = 'super-resolution-model-tests/transformer/generated_image/'
        os.makedirs(self.image_dir, exist_ok=True)

    def compute_psnr(self, img1, img2, max_val=1.0):
        mse = F.mse_loss(img1, img2)
        psnr = 10 * torch.log10((max_val ** 2) / mse)
        return psnr

    def loopEpochs(self, num_epoch):
        self.model.train()
        scaler = torch.amp.GradScaler('cuda')
        timesNotImproved = 0
        for epoch in range(self.epochs):
            start_time = time.time()
            total_loss = 0

            for idx, (lr_img, hr_img) in enumerate(self.train_loader):
                lr_img = lr_img.to(self.device)
                hr_img = hr_img.to(self.device)

                self.optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    sr_img = self.model(lr_img)
                    loss = self.loss_func(sr_img, hr_img)

                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

                scaler.step(self.optimizer)
                scaler.update()
                self.scheduler.step()
                total_loss += loss.item()


            # losses once per epoch
            print(f'Epoch [{epoch + 1}/{num_epoch}]  | Loss {total_loss/self.train_loader.dataset.__len__():.4f} | Time: {time.time() - start_time:.2f} sec')

            # Visualise the generated image at different epochs
            if (epoch + 1) % 20 == 0:
                self.visualise_validation_set(epoch)
                    


    def visualise_validation_set(self, epoch):
        self.model.eval()

        total_psnr = 0.0
        count = 0
        imageChosenNum = random.randint(0, self.val_loader.dataset.__len__() - 1)
        imageChosen = None
        scale_factor = self.val_loader.dataset.scale

        # Loop over the entire validation loader

        for idx, (lr_img_full, hr_img_full) in enumerate(self.val_loader):
            
            _, _, H_lr, W_lr = lr_img_full.shape
            
            TILE_SIZE = 128
            OVERLAP = 32  # Define overlap amount (e.g., 25% of tile size)
            STRIDE = TILE_SIZE - OVERLAP

            # Create tensors to store the stitched image and a counter
            H_sr = H_lr * scale_factor
            W_sr = W_lr * scale_factor
            sr_img_stitched = torch.zeros(hr_img_full.shape[1], H_sr, W_sr, dtype=hr_img_full.dtype).to(self.device)
            stitch_counter = torch.zeros(hr_img_full.shape[1], H_sr, W_sr, dtype=hr_img_full.dtype).to(self.device)

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
                        lr_patch = lr_img_full[:, :, h_start_actual:h_end, w_start_actual:w_end].to(self.device)
                    
                        # Generate SR patch
                        sr_patch = self.model(lr_patch).squeeze(0) # Keep on device for now

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
            psnr_sr = self.compute_psnr(hr_img_vis, sr_img_vis).item()
        
            # Accumulate PSNR
            total_psnr += psnr_sr
            count += 1

            # Grid
            if idx == imageChosenNum:
                lr_resized_vis = F.interpolate(lr_img_full, 
                                               scale_factor=scale_factor, 
                                               mode='nearest').squeeze(0).clamp(0, 1)

                comparison = torch.stack([lr_resized_vis, sr_img_vis, hr_img_vis], dim=0)
                grid = torchvision.utils.make_grid(comparison, nrow=3, value_range=(0,1))
                grid_np = grid.permute(1,2,0).numpy()
                first_image_grid = (grid_np * 255).astype(np.uint8)

        avg_psnr = total_psnr / count if count > 0 else 0.0

        # Save Checkpoint
        if self.best_psnr < avg_psnr:
            self.best_psnr = avg_psnr
            save_checkpoint(epoch, avg_psnr, self.model, self.checkpoint_dir)
            alert.send_notification(f"New best AVG PSNR: Epoch {epoch + 1}  PSNR: {avg_psnr:.2f}")
            filename = f"{self.image_dir}/top_image.png"
            Image.fromarray(first_image_grid).save(filename)

        # Print Final PSNR value
        print(f'Validation Average PSNR (Full Image Tiling): {avg_psnr:.4f}')

        self.model.train()

if __name__ == "__main__":
    main(parser.parse_args())