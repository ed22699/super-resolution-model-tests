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
from diffusion import Diffusion,  SR3_UNet
from utils.save_checkpoint_diff import save_checkpoint
import torchvision.transforms.functional as TF
import random

# Alert messages
import alert


def main():

    config = {
        'dataset': 'DIV2K',
        'img_size': (640, 640, 3),
        'timestep_embedding_dim': 256,
        'n_layers': 8,
        'hidden_dim': 256,
        'n_timesteps': 1000,
        'train_batch_size': 128,
        'inference_batch_size': 64,
        'lr': 1e-5,
        'epochs': 1000,
        'seed': 42,
    }

    hidden_dims = [config['hidden_dim'] for _ in range(config['n_layers'])]
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # For GPU (CUDA) or CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess the dataset
    transformList = []
    transformList.append(transforms.ToTensor())
    basic_transforms = transforms.Compose(transformList)

    datasetRoot = "DIV2K"
    lr_path = datasetRoot+"/DIV2K_train_LR_x8"
    hr_path = datasetRoot+"/DIV2K_train_HR"

    train_dataset = GANDIV2KDataLoader(
        root_dir_lr=lr_path,
        root_dir_hr=hr_path,
        transform=basic_transforms,
        mode="train",
        batch_size=256,
        scale=8,
        patch_size=80,
    )

    lr_path = datasetRoot+"/DIV2K_valid_LR_x8"
    hr_path = datasetRoot+"/DIV2K_valid_HR"

    val_dataset = GANDIV2KDataLoader(
        root_dir_lr=lr_path,
        root_dir_hr=hr_path,
        transform=basic_transforms,
        mode="val",
        batch_size=64,
        scale=8,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=8,
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

    model = SR3_UNet(in_channels = 3,
                     out_channels = 3,
                     inner_channel = config['hidden_dim'],
                     time_emb_dim = config['timestep_embedding_dim']
                     ).to(device)
    diffusion = Diffusion(model, image_resolution=config['img_size'],  
                          n_times=config['n_timesteps'],
                          device=device
                          ).to(device)
    optimizer = optim.Adam(diffusion.parameters(), lr = config['lr'])

    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    trainer = Trainer(loss_func, diffusion, train_loader, val_loader, optimizer, scheduler, device)
    trainer.loopEpochs(config['epochs'])

    alert.send_notification(f"Training finished for visual AI")

        

class Trainer:
    def __init__(self, loss_func, model, train_loader, val_loader, optimizer, scheduler, device):
        self.best_psnr = 0.0
        self.loss_func = loss_func
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

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

        for epoch in range(num_epoch):
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            start_time = time.time()
            total_loss = 0

            for idx, (lr_img, hr_img) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                lr_img = lr_img.to(self.device)
                hr_img = hr_img.to(self.device)
                with torch.amp.autocast('cuda'):
                    noisy_hr, epsilon, pred_epsilon = self.model(lr_img, hr_img)
                    loss = self.loss_func(pred_epsilon, epsilon)
                total_loss += loss
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                total_loss += loss.item()
                torch.cuda.empty_cache()

            # losses once per epoch
            print(f'Epoch [{epoch + 1}/{num_epoch}]  | Loss {total_loss} | Time: {time.time() - start_time:.2f} sec')


            # Visualise the generated image at different epochs
            if (epoch + 1) % 20 == 0:
                # self.visualise_generated_images(epoch)
                self.visualise_validation_set(epoch)

    def visualise_validation_set(self, epoch):
        self.model.eval()

        total_psnr = 0.0
        count = 0
        imageChosen = None
        scale_factor = self.val_loader.dataset.scale
        num_images = self.val_loader.dataset.__len__() // 5 # only check on a 5th of the set
        imageChosenNum = random.randint(0, num_images - 1)

        # Loop over the entire validation loader
        for idx, (lr_img_full, hr_img_full) in enumerate(self.val_loader):
            lr_img_full = lr_img_full.to(self.device)
            hr_img_full = hr_img_full.to(self.device)
            
            with torch.no_grad():
                sr_img = self.model.speedSample(lr_img_full).to(self.device).squeeze(0) 
                
            hr_img_vis = hr_img_full.squeeze(0).clamp(0, 1)
            sr_img_vis = sr_img.clamp(0, 1)

            # Align sizing issues from patching
            H = min(hr_img_vis.shape[1], sr_img_vis.shape[1])
            W = min(hr_img_vis.shape[2], sr_img_vis.shape[2])
            hr_img_vis = hr_img_vis[:, :H, :W]
            sr_img_vis = sr_img_vis[:, :H, :W]

            # Calculate PSNR for this image
            mse_sr = self.compute_mse(hr_img_vis, sr_img_vis).item()
            psnr_sr = self.compute_psnr(mse_sr).item()
        
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
                grid_np = grid.permute(1,2,0).cpu().numpy()
                first_image_grid = (grid_np * 255).astype(np.uint8)

            if num_images == idx:
                break

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
    main()
