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

def discriminator_loss(real_output, fake_output):
    # Loss for real images
    real_loss = loss_func(real_output, torch.ones_like(real_output).to(device))
    # Loss for fake images
    fake_loss = loss_func(fake_output, torch.zeros_like(fake_output).to(device))

    loss_D = real_loss + fake_loss
    return loss_D

def generator_loss(fake_output):
    # Compare discriminator's output on fake images with target labels of 1
    loss_G = loss_func(fake_output, torch.ones_like(fake_output).to(device))
    return loss_G

def training(lr_img, hr_img):

    '''Training step for the Discriminator'''
    real_x = hr_img.to(device)
    real_output = gan_D(real_x)
    fake_x = gan_G(lr_img.to(device)).detach()
    fake_output = gan_D(fake_x)
    loss_D =discriminator_loss(real_output, fake_output)

    # Backpropagate the discriminator loss and update its parameters
    optim_D.zero_grad()
    loss_D.backward()
    optim_D.step()

    '''Training step for the Generator'''
    fake_x = gan_G(lr_img.to(device))
    fake_output = gan_D(fake_x)
    loss_G = generator_loss(fake_output)

    # Backpropagate the generator loss and update its parameters
    optim_G.zero_grad()
    loss_G.backward()
    optim_G.step()

    return loss_D, loss_G

def visualise_generated_images(generator, epoch, image_dir, num_samples=1):
    generator.eval()

    for i in range(num_samples):
        # Pick a sample
        lr_img, hr_img = train_dataset[i]
        lr_img = lr_img.unsqueeze(0).to(device)
        hr_img = hr_img.to(device)

        # # --- Resize HR and LR to match SR size for visualization ---
        lr_resized = F.interpolate(lr_img, scale_factor=8 , mode="nearest")
        lr_resized = lr_resized.squeeze(0)


        # # --- Make comparison grid ---
        with torch.no_grad():
            sr_img = generator(lr_img).squeeze(0)   # [3,H,W], values likely [-1,1]

        # # Map to 0-1
        # sr_img_vis = (sr_img + 1) / 2
        # sr_img_vis = sr_img_vis.clamp(0,1)

        # # Also make LR/HR consistent
        # lr_resized_vis = lr_resized.clamp(0,1)
        # hr_img_vis = hr_img.squeeze(0).clamp(0,1)

        # # Stack for grid
        # comparison = torch.stack([lr_resized_vis, sr_img_vis, hr_img_vis], dim=0)

        # grid = torchvision.utils.make_grid(comparison, nrow=3, value_range=(0,1))
        # grid_np = grid.permute(1,2,0).cpu().numpy()
        # grid_uint8 = (grid_np * 255).astype(np.uint8)


        # # --- Save ---
        # filename = f"{image_dir}/epoch_{epoch}_sample_{i}.png"
        # Image.fromarray(grid_uint8).save(filename)
        comparison = torch.stack([lr_resized, sr_img, hr_img], dim=0)

        filename = f"{image_dir}/epoch_{epoch}_sample_{i}.png"
        
        # Use torchvision's specialized function. 
        # It handles clamping (0-1), numpy conversion, and saving correctly.
        # Normalize=False means we trust the values are already in the display range (0-1).
        torchvision.utils.save_image(
            comparison, 
            filename, 
            nrow=3, 
            normalize=False, 
            value_range=(0, 1) # Ensure the output is mapped correctly to 0-255
        )

    generator.train()

def visualise_loss(losses_D, losses_G, image_dir, loss_type):
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

input_dim = 3
num_epoch = 50
checkpoint_dir = 'super-resolution-model-tests/training_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
image_dir = 'super-resolution-model-tests/generated_image/'
os.makedirs(image_dir, exist_ok=True)

gan_G = Generator(input_dim).to(device)
gan_D = Discriminator().to(device)

# Define separate Adam optimizers for generator and discriminator
optim_G = torch.optim.Adam(gan_G.parameters(), lr=0.0002)
optim_D = torch.optim.Adam(gan_D.parameters(), lr=0.0002)

# Initialise the list to store the losses for each epoch
iteration_losses_D = []
iteration_losses_G = []
epoch_losses_D = []
epoch_losses_G = []

for epoch in range(num_epoch):
    start_time = time.time()
    total_loss_D, total_loss_G = 0, 0

    for idx, (lr_img, hr_img) in enumerate(train_loader):
        loss_D, loss_G = training(lr_img, hr_img)

        iteration_losses_D.append(loss_D.detach().item())
        iteration_losses_G.append(loss_G.detach().item())
        total_loss_D += loss_D.detach().item()
        total_loss_G += loss_G.detach().item()

    epoch_losses_D.append(total_loss_D / len(train_loader))
    epoch_losses_G.append(total_loss_G / len(train_loader))

    # Save model checkpoints
    if (epoch + 1) % 10 == 0:
        save_checkpoint(epoch + 1, gan_G, gan_D, optim_G, optim_D, checkpoint_dir)

    # losses once per epoch
    print(f'Epoch [{epoch + 1}/{num_epoch}] | Loss_D {iteration_losses_D[-1]:.4f} | Loss_G {iteration_losses_G[-1]:.4f} | Time: {time.time() - start_time:.2f} sec')
    print(f'Epoch [{epoch + 1}/{num_epoch}]  | Loss_D {epoch_losses_D[epoch]:.4f} | Loss_G {epoch_losses_G[epoch]:.4f} | Time: {time.time() - start_time:.2f} sec')

    # Task1: visualise the generated image at different epochs
    visualise_generated_images(gan_G, epoch, image_dir)

# Task2: visualise the loss through a plot
visualise_loss(iteration_losses_D, iteration_losses_G, image_dir, 'Iteration')
visualise_loss(epoch_losses_D, epoch_losses_G, image_dir, 'Epoch')