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
from gan import Generator, Discriminator, VGGFeatureExtractor
from utils.save_checkpoint import save_checkpoint
import torchvision.transforms.functional as TF
import random

# Alert messages
import alert


def main():
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
        batch_size=64,
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
    loss_func = nn.BCEWithLogitsLoss()
    input_dim = 3
    num_epoch = 500

    gan_G = Generator(input_dim).to(device)
    gan_D = Discriminator().to(device)

    # define separate Adam optimizers
    # do 0.0002 (1.64), 0.00015 (1.79), 0.00025 (1.31), 0.0003 (1.16), 0.00035 for G
    optim_G = optim.Adam(gan_G.parameters(), lr=0.00035) 
    # was 0.00002
    optim_D = optim.Adam(gan_D.parameters(), lr=0.00002)

    # learning rate scheduler
    scheduler_G = torch.optim.lr_scheduler.StepLR(optim_G, step_size=10, gamma=0.5)
    # scheduler_D = torch.optim.lr_scheduler.StepLR(optim_D, step_size=10, gamma=0.5)

    trainer = Trainer(loss_func, gan_G, gan_D, train_loader, val_loader, optim_G, optim_D, scheduler_G, device)
    trainer.loopEpochs(num_epoch)

    alert.send_notification(f"Training finished for visual AI")

        

class Trainer:
    def __init__(self, loss_func, gan_G, gan_D, train_loader, val_loader, optim_G, optim_D, scheduler_G, device):
        self.best_psnr = 0.0
        self.loss_func = loss_func
        self.gan_G = gan_G
        self.gan_D = gan_D
        self.optim_G = optim_G
        self.optim_D = optim_D
        self.device = device
        # self.scheduler_D = scheduler_D
        self.scheduler_G = scheduler_G
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Directories
        self.checkpoint_dir = 'super-resolution-model-tests/training_checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.image_dir = 'super-resolution-model-tests/generated_image/'
        os.makedirs(self.image_dir, exist_ok=True)

        # Loss
        self.iteration_losses_D = []
        self.iteration_losses_G = []
        self.epoch_losses_D = []
        self.epoch_losses_G = []
        self.vgg_extractor = VGGFeatureExtractor().to(self.device)
        self.lambda_vgg = 0.4
        self.l1_loss = nn.L1Loss()

    def compute_mse(self, img1, img2):
        return F.mse_loss(img1, img2)

    def compute_psnr(self, mse):
        mse_tensor = torch.tensor(mse) 
        return 10 * torch.log10(1 / mse_tensor)

    def discriminator_loss(self, real_output, fake_output):
        # Loss for real images
        real_loss = self.loss_func(real_output, torch.full_like(real_output, 0.8).to(self.device))
        # Loss for fake images
        fake_loss = self.loss_func(fake_output, torch.full_like(fake_output, 0.2).to(self.device))

        loss_D = real_loss + fake_loss
        return loss_D

    def generator_loss(self, fake_output, fake_x, hr_img):
        # L_Adversarial
        loss_adv = self.loss_func(fake_output, torch.full_like(fake_output, 0.8).to(self.device))
        
        # L_Pixel, fake_x is the SR image, hr_img is the ground truth
        loss_pixel = self.l1_loss(fake_x, hr_img)
        
        # L_Perceptual
        VGG_SCALE_FACTOR = 0.5
        fake_x_vgg = F.interpolate(fake_x, scale_factor=VGG_SCALE_FACTOR, mode='bilinear', align_corners=False)
        hr_img_vgg = F.interpolate(hr_img, scale_factor=VGG_SCALE_FACTOR, mode='bilinear', align_corners=False)
        vgg_fake = self.vgg_extractor(fake_x_vgg)
        vgg_real = self.vgg_extractor(hr_img_vgg).detach()
        
        # L1 Loss on the feature maps
        loss_perceptual = self.l1_loss(vgg_fake, vgg_real)
        
        # Combined Generator Loss
        lambda_pixel = 0.1
        
        loss_G = loss_adv + (self.lambda_vgg * loss_perceptual) + (lambda_pixel * loss_pixel)
        return loss_G

    def loopEpochs(self, num_epoch):
        for epoch in range(num_epoch):
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            start_time = time.time()
            total_loss_D, total_loss_G = 0, 0

            for idx, (lr_img, hr_img) in enumerate(self.train_loader):
                loss_D, loss_G = self.training(lr_img, hr_img)

                self.iteration_losses_D.append(loss_D.detach().item())
                self.iteration_losses_G.append(loss_G.detach().item())
                total_loss_D += loss_D.detach().item()
                total_loss_G += loss_G.detach().item()

            self.epoch_losses_D.append(total_loss_D / len(self.train_loader))
            self.epoch_losses_G.append(total_loss_G / len(self.train_loader))


            # Learning rate updated
            self.scheduler_G.step()
            # self.scheduler_D.step()


            # losses once per epoch
            print(f'Epoch [{epoch + 1}/{num_epoch}]  | Loss_D {self.epoch_losses_D[epoch]:.4f} | Loss_G {self.epoch_losses_G[epoch]:.4f} | Time: {time.time() - start_time:.2f} sec')

            # Visualise the generated image at different epochs
            if (epoch + 1) % 5 == 0:
                # self.visualise_generated_images(epoch)
                self.visualise_validation_set(epoch)

        # Visualise the loss through a plot
        self.visualise_loss(self.iteration_losses_D, self.iteration_losses_G, self.image_dir, 'Iteration')
        self.visualise_loss(self.epoch_losses_D, self.epoch_losses_G, self.image_dir, 'Epoch')
        
    def training(self, lr_img, hr_img):
        # Training step for the Discriminator
        real_x = hr_img.to(self.device)
        real_output = self.gan_D(real_x)
        fake_x = self.gan_G(lr_img.to(self.device)).detach()
        fake_output = self.gan_D(fake_x)
        loss_D =self.discriminator_loss(real_output, fake_output)

        # Backpropagate the discriminator loss and update its parameters
        self.optim_D.zero_grad()
        loss_D.backward()
        self.optim_D.step()

        # Training step for the Generator
        fake_x = self.gan_G(lr_img.to(self.device))
        fake_output = self.gan_D(fake_x)
        loss_G = self.generator_loss(fake_output, fake_x, hr_img.to(self.device))

        # Backpropagate the generator loss and update its parameters
        self.optim_G.zero_grad()
        loss_G.backward()
        self.optim_G.step()

        return loss_D, loss_G

    def visualise_validation_set(self, epoch):
        self.gan_G.eval()

        total_psnr = 0.0
        count = 0
        imageChosenNum = random.randint(0, self.val_loader.dataset.__len__() - 1)
        imageChosen = None
        scale_factor = self.val_loader.dataset.scale

        # Loop over the entire validation loader
        for idx, (lr_img_full, hr_img_full) in enumerate(self.val_loader):
        
            _, _, H_lr, W_lr = lr_img_full.shape
        
            TILE_SIZE = 128

            # Calculate number of tiles needed
            num_h = H_lr // TILE_SIZE + (1 if H_lr % TILE_SIZE != 0 else 0)
            num_w = W_lr // TILE_SIZE + (1 if W_lr % TILE_SIZE != 0 else 0)

            # Create an empty tensor to store the stitched SR image
            H_sr = H_lr * scale_factor
            W_sr = W_lr * scale_factor
            sr_img_stitched = torch.zeros(hr_img_full.shape[1], H_sr, W_sr, dtype=hr_img_full.dtype)

            # Tiling and Stitching Loop
            with torch.no_grad():
                for i in range(num_h):
                    for j in range(num_w):
                        # Patch
                        start_h_lr = i * TILE_SIZE
                        end_h_lr = min((i + 1) * TILE_SIZE, H_lr)
                        start_w_lr = j * TILE_SIZE
                        end_w_lr = min((j + 1) * TILE_SIZE, W_lr)
                    
                        lr_patch = lr_img_full[:, :, start_h_lr:end_h_lr, start_w_lr:end_w_lr].to(self.device)
                    
                        # Generate
                        sr_patch = self.gan_G(lr_patch).cpu().squeeze(0)

                        # Stitch
                        start_h_sr = start_h_lr * scale_factor
                        end_h_sr = end_h_lr * scale_factor
                        start_w_sr = start_w_lr * scale_factor
                        end_w_sr = end_w_lr * scale_factor

                        sr_img_stitched[:, start_h_sr:end_h_sr, start_w_sr:end_w_sr] = sr_patch

            hr_img_vis = hr_img_full.squeeze(0).clamp(0, 1)
            sr_img_vis = sr_img_stitched.clamp(0, 1)

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
                grid_np = grid.permute(1,2,0).numpy()
                first_image_grid = (grid_np * 255).astype(np.uint8)

        avg_psnr = total_psnr / count if count > 0 else 0.0

        # Save Checkpoint
        if self.best_psnr < avg_psnr:
            self.best_psnr = avg_psnr
            save_checkpoint(epoch, avg_psnr, self.gan_G, self.gan_D, self.checkpoint_dir)
            alert.send_notification(f"New best AVG PSNR: Epoch {epoch + 1}  PSNR: {avg_psnr:.2f}")
            filename = f"{self.image_dir}/top_image.png"
            Image.fromarray(first_image_grid).save(filename)

        # Print Final PSNR value
        print(f'Validation Average PSNR (Full Image Tiling): {avg_psnr:.4f}')

        self.gan_G.train()

    def visualise_loss(self, losses_D, losses_G, image_dir, loss_type):
        plt.figure(figsize=(10, 5))
        plt.plot(losses_D, label='Discriminator Loss')
        plt.plot(losses_G, label='Generator Loss')
        plt.title('Training Loss')
        plt.xlabel(f'{loss_type}')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.grid()
        plt.savefig(f'{image_dir}/training_loss_{loss_type}.png')  # save the loss plot
        plt.show()
        plt.close()

if __name__ == "__main__":
    main()
