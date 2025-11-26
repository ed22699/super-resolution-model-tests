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
from gan import Generator, Discriminator
from utils.save_checkpoint import save_checkpoint
import torchvision.transforms.functional as TF


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
        batch_size=256,
        scale=8
    )

    lr_path = datasetRoot+"/DIV2K_valid_LR_x8"
    hr_path = datasetRoot+"/DIV2K_valid_HR"

    val_dataset = GANDIV2KDataLoader(
        root_dir_lr=lr_path,
        root_dir_hr=hr_path,
        transform=basic_transforms,
        mode="val",
        batch_size=256,
        scale=8
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
        batch_size=4,
        num_workers=cpu_count(),
        pin_memory=True,
    )

    # Define the Binary Cross Entropy loss function
    loss_func = nn.BCEWithLogitsLoss()
    input_dim = 3
    num_epoch = 50

    gan_G = Generator(input_dim).to(device)
    gan_D = Discriminator().to(device)

    # define separate Adam optimizers
    lr = 0.0001
    optim_G = optim.Adam(gan_G.parameters(), lr=lr)
    optim_D = optim.Adam(gan_D.parameters(), lr=lr)

    # learning rate scheduler
    scheduler_G = torch.optim.lr_scheduler.StepLR(optim_G, step_size=5, gamma=0.5)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optim_D, step_size=5, gamma=0.5)

    trainer = Trainer(loss_func, gan_G, gan_D, train_loader, val_loader, optim_G, optim_D, scheduler_D, scheduler_G, device)
    trainer.loopEpochs(num_epoch)

        

class Trainer:
    def __init__(self, loss_func, gan_G, gan_D, train_loader, val_loader, optim_G, optim_D, scheduler_D, scheduler_G, device):
        self.best_psnr = 0.0
        self.loss_func = loss_func
        self.gan_G = gan_G
        self.gan_D = gan_D
        self.optim_G = optim_G
        self.optim_D = optim_D
        self.device = device
        self.scheduler_D = scheduler_D
        self.scheduler_G = scheduler_G
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.iteration_losses_D = []
        self.iteration_losses_G = []
        self.epoch_losses_D = []
        self.epoch_losses_G = []
        self.checkpoint_dir = 'super-resolution-model-tests/training_checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.image_dir = 'super-resolution-model-tests/generated_image/'
        os.makedirs(self.image_dir, exist_ok=True)

    def compute_mse(self, img1, img2):
        return F.mse_loss(img1, img2)

    def compute_psnr(self, mse):
        mse_tensor = torch.tensor(mse) 
        return 10 * torch.log10(1 / mse_tensor)

    def discriminator_loss(self, real_output, fake_output):
        # Loss for real images
        real_loss = self.loss_func(real_output, torch.ones_like(real_output).to(self.device))
        # Loss for fake images
        fake_loss = self.loss_func(fake_output, torch.zeros_like(fake_output).to(self.device))

        loss_D = real_loss + fake_loss
        return loss_D

    def generator_loss(self, fake_output):
        # Compare discriminator's output on fake images with target labels of 1
        loss_G = self.loss_func(fake_output, torch.ones_like(fake_output).to(self.device))
        return loss_G

    def loopEpochs(self, num_epoch):
        for epoch in range(num_epoch):
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
            self.scheduler_D.step()


            # losses once per epoch
            print(f'Epoch [{epoch + 1}/{num_epoch}]  | Loss_D {self.epoch_losses_D[epoch]:.4f} | Loss_G {self.epoch_losses_G[epoch]:.4f} | Time: {time.time() - start_time:.2f} sec')

            # Task1: visualise the generated image at different epochs
            if (epoch + 1) % 5 == 0:
                self.visualise_generated_images(epoch)

        # Task2: visualise the loss through a plot
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
        loss_G = self.generator_loss(fake_output)

        # Backpropagate the generator loss and update its parameters
        self.optim_G.zero_grad()
        loss_G.backward()
        self.optim_G.step()

        return loss_D, loss_G

    def visualise_generated_images(self, epoch, num_samples=1):
        self.gan_G.eval()

        for i in range(num_samples):
            # Pick a sample
            lr_img, hr_img = self.train_dataset[i]
            lr_img = lr_img.unsqueeze(0).to(self.device)
            hr_img = hr_img.to(self.device)

            # Resize HR and LR to match SR size for visualization
            lr_resized = F.interpolate(lr_img, scale_factor=8 , mode="nearest")
            lr_resized = lr_resized.squeeze(0)

            # Make comparison grid
            with torch.no_grad():
                sr_img = self.gan_G(lr_img).squeeze(0)

            # Map to 0-1
            sr_min = sr_img.min()
            sr_max = sr_img.max()
            sr_vis = (sr_img - sr_min) / (sr_max - sr_min + 1e-5)
            sr_img_vis = sr_vis.clamp(0,1)

            # Also make LR/HR consistent
            lr_resized_vis = lr_resized.clamp(0,1)
            hr_img_vis = hr_img.squeeze(0).clamp(0,1)

            # Stack for grid
            comparison = torch.stack([lr_resized_vis, sr_img_vis, hr_img_vis], dim=0)

            grid = torchvision.utils.make_grid(comparison, nrow=3, value_range=(0,1))
            grid_np = grid.permute(1,2,0).cpu().numpy()
            grid_uint8 = (grid_np * 255).astype(np.uint8)


        mse_sr = self.compute_mse(hr_img_vis, sr_img_vis).item()
        psnr_sr = self.compute_psnr(mse_sr)
        # --- Save ---
        if self.best_psnr < psnr_sr:
            self.best_psnr = psnr_sr
            filename = f"{self.image_dir}/top_image.png"
            Image.fromarray(grid_uint8).save(filename)
            save_checkpoint(epoch, psnr, self.gan_G, self.gan_D, self.optim_G, self.optim_D, self.checkpoint_dir)

        # Print MSE and PSNR values
        print(f'MSE and PSNR for SR image: MSE = {mse_sr}, PSNR = {psnr_sr}')

        self.gan_G.train()

    def visualise_loss(self, losses_D, losses_G, image_dir, loss_type):
        plt.plot(losses_D, label="losses_D")
        plt.plot(losses_G, label="losses_G")
        plt.title(loss_type)
        plt.legend()
        plt.savefig(f"./{image_dir}/loss_type: {loss_type}.png")
        plt.show()    
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
