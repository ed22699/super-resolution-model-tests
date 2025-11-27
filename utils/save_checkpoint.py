import os
import torch

def save_checkpoint(epoch, psnr, generator, discriminator, checkpoint_dir):
    # Save the model and optimizer states
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
    }
    # Save the checkpoint
    for filename in os.listdir(checkpoint_dir):
        file_path = os.path.join(checkpoint_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(filename, "is removed")
    torch.save(checkpoint, os.path.join(checkpoint_dir, f'ckpt_PSNR_{psnr:.4f}.pth'))
    print(f'Checkpoint saved for epoch {epoch}')
