import os
import torch

def save_checkpoint(epoch, generator, discriminator, generator_optimizer, discriminator_optimizer, checkpoint_dir):
    # Save the model and optimizer states
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'generator_optimizer_state_dict': generator_optimizer.state_dict(),
        'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict()
    }
    # Save the checkpoint
    torch.save(checkpoint, os.path.join(checkpoint_dir, f'ckpt_epoch_{epoch:04d}.pth'))
    print(f'Checkpoint saved for epoch {epoch}')
