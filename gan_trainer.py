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

def compute_mse(img1, img2):
    return F.mse_loss(img1, img2)
def compute_psnr(mse):
    mse_tensor = torch.tensor(mse)  # Convert the float to a tensor
    return 10 * torch.log10(1 / mse_tensor)

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

        # --- Resize HR and LR to match SR size for visualization ---
        lr_resized = F.interpolate(lr_img, scale_factor=8 , mode="nearest")
        lr_resized = lr_resized.squeeze(0)

        # --- Make comparison grid ---
        with torch.no_grad():
            sr_img = generator(lr_img).squeeze(0)   # [3,H,W], values likely [-1,1]

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

        # --- Save ---
        filename = f"{image_dir}/epoch_{epoch}_sample_{i}.png"
        Image.fromarray(grid_uint8).save(filename)

    mse_sr = compute_mse(hr_img_vis, sr_img_vis).item()
    psnr_sr = compute_psnr(mse_sr)

    # Print MSE and PSNR values
    print(f'MSE and PSNR for SR image: MSE = {mse_sr}, PSNR = {psnr_sr}')

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

# define separate Adam optimizers
lr = 0.0001
optim_G = optim.Adam(gan_G.parameters(), lr=lr)
optim_D = optim.Adam(gan_D.parameters(), lr=lr)

# learning rate scheduler
scheduler_G = torch.optim.lr_scheduler.StepLR(optim_G, step_size=5, gamma=0.5)
scheduler_D = torch.optim.lr_scheduler.StepLR(optim_D, step_size=5, gamma=0.5)

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

    # Learning rate updated
    scheduler_G.step()
    scheduler_D.step()


    # losses once per epoch
    print(f'Epoch [{epoch + 1}/{num_epoch}]  | Loss_D {epoch_losses_D[epoch]:.4f} | Loss_G {epoch_losses_G[epoch]:.4f} | Time: {time.time() - start_time:.2f} sec')

    # Task1: visualise the generated image at different epochs
    if (epoch + 1) % 10 == 0:
        visualise_generated_images(gan_G, epoch, image_dir)

# Task2: visualise the loss through a plot
visualise_loss(iteration_losses_D, iteration_losses_G, image_dir, 'Iteration')
visualise_loss(epoch_losses_D, epoch_losses_G, image_dir, 'Epoch')

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
# import os

# from utils.GAN.generator import Generator
# from utils.GAN.discriminator import Discriminator
# from utils.GAN.loss import discriminator_loss, generator_loss
# from utils.GAN.weights_init import weights_init

### Challenge - Task3: MSE and PSNR computation for the denoised image and the original image
### Challenge - Task4: MSE and PSNR computation for the noisy image and the original image
# def compute_mse(img1, img2):
#     return F.mse_loss(img1, img2)
# def compute_psnr(mse):
#     mse_tensor = torch.tensor(mse)  # Convert the float to a tensor
#     return 10 * torch.log10(1 / mse_tensor)

# def training(lr_img, hr_img):

#     '''Training step for the Discriminator'''
#     real_x = hr_img.to(device)
#     real_output = gan_D(real_x)
#     fake_x = gan_G(lr_img.to(device)).detach()
#     fake_output = gan_D(fake_x)
#     loss_D =discriminator_loss(real_output, fake_output)

#     # Backpropagate the discriminator loss and update its parameters
#     optim_D.zero_grad()
#     loss_D.backward()
#     optim_D.step()

#     '''Training step for the Generator'''
#     fake_x = gan_G(lr_img.to(device))
#     fake_output = gan_D(fake_x)
#     loss_G = generator_loss(fake_output)

#     # Backpropagate the generator loss and update its parameters
#     optim_G.zero_grad()
#     loss_G.backward()
#     optim_G.step()

#     return loss_D, loss_G



#     return loss_D.item(), loss_G.item(), noisy_imgs, clean_imgs


# # Main function
# if __name__ == "__main__":


#     # # save Directory
#     # checkpoint_dir = './training_checkpoints'
#     # os.makedirs(checkpoint_dir, exist_ok=True)
#     # image_dir = './denoised_images'
#     # os.makedirs(image_dir, exist_ok=True)

#     # # Initialisation of the network weights
#     # # generator.apply(weights_init)
#     # # discriminator.apply(weights_init)

#     # # Step 4 - training loop
#     # # def training_loop(clean_imgs)

#     # # Step 5 - train the model
#     # num_epochs = 10
#     # for epoch in range(num_epochs):
#     #     total_loss_D, total_loss_G = 0, 0
#     #     for batch_idx, (lr_img, hr_img) in enumerate(train_loader):
#     #         # Get noisy_imgs and clean_imgs
#     #         loss_D, loss_G, sr_imgs, hr_imgs = training_loop(lr_img, hr_img)
#     #         total_loss_D += loss_D
#     #         total_loss_G += loss_G

#     #         # Save only the first batch of images for visualisation
#     #         if batch_idx == 0:
#     #             # Save noisy images
#     #             sr_img_np = sr_imgs[0].cpu().numpy()
#     #             plt.imshow(sr_img_np[0], cmap='gray')  # (channels, height, width) from torch, only 1 channel, because the image is greyscale
#     #             plt.title('SR Image')
#     #             plt.savefig(f'{image_dir}/epoch_{epoch + 1}_sample.png')
#     #             plt.close()

#     #             # Save denoised images
#     #             sr_img = generator(sr_imgs.view(sr_imgs.size(0), -1)).view(sr_imgs.size(0), 1, 28, 28)
#     #             sr_img = torch.clamp(sr_img, 0., 1.)
#     #             sr_img_np = denoised_img[0].detach().cpu().numpy()
#     #             plt.imshow(denoised_img_np[0], cmap='gray')
#     #             plt.title('Denoised Image')
#     #             plt.savefig(f'{image_dir}/epoch_{epoch + 1}_denoised.png')
#     #             plt.close()

#     #             # Save clean images
#     #             clean_img_np = clean_imgs[0].cpu().numpy()
#     #             plt.imshow(clean_img_np[0], cmap='gray')
#     #             plt.title('Clean Image')
#     #             plt.savefig(f'{image_dir}/epoch_{epoch + 1}_clean.png')
#     #             plt.close()

#     #             # Challenge - Compute MSE and PSNR for the noisy and denoised images
#     #             mse_noisy = compute_mse(noisy_imgs, clean_imgs).item()
#     #             mse_denoised = compute_mse(denoised_img, clean_imgs).item()
#     #             psnr_noisy = compute_psnr(mse_noisy)
#     #             psnr_denoised = compute_psnr(mse_denoised)

#     #             # Print MSE and PSNR values
#     #             print(f'Epoch {epoch + 1}/{num_epochs}, MSE and PSNR for noisy and original images: MSE = {mse_noisy}, PSNR = {psnr_noisy}')
#     #             print(f'Epoch {epoch + 1}/{num_epochs}, MSE and PSNR for the denoised and original images: MSE = {mse_denoised}, PSNR = {psnr_denoised}')

#     #     # Learning rate updated
#     #     scheduler_G.step()
#     #     scheduler_D.step()

#     #     # Print the average loss per epoch
#     #     print(f'After Epoch [{epoch + 1}/{num_epochs}], D_loss: {total_loss_D / len(train_loader):.4f}, G_loss: {total_loss_G / len(train_loader):.4f}')

#     # print("Training completed!")
