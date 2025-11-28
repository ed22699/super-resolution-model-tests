import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPosEmb(nn.Module):
    """
    Generates embeddings for time steps to help the model recognise the
    relative positions of each diffusion step.

    Attributes
    ----------
    dim : int
        Specifies the dimension of the embedding.
    """

    def __init__(self, dim):
        """
        Generates embeddings for time steps to help the model recognise the
        relative positions of each diffusion step.

        Parameters
        ----------
        dim : int
            Specifies the dimension of the embedding.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        Creates position-aware embeddings for a batch of time steps x

        Parameters
        ----------
        x : int
            Batch of time steps

        Returns
        ----------
        emb : int
            Embedded batch of time steps
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResBlock(nn.Module):
    # A simplified Residual Block, often adapted from BigGAN or DDPM
    def __init__(self, in_c, out_c, time_emb_dim, norm_groups=32):
        super().__init__()
        # Time embedding (FiLM modulation)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 2 * out_c) # Output two vectors for scale/shift
        )
        # Block layers
        self.block = nn.Sequential(
            nn.GroupNorm(norm_groups, in_c),
            nn.SiLU(),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups, out_c),
            nn.SiLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        )
        self.res_conv = nn.Conv2d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()

    def forward(self, x, time_emb):
        scale, shift = self.mlp(time_emb).chunk(2, dim=1) # Split into two for FiLM
        
        h = self.block(x)
        # FiLM (Feature-wise Linear Modulation)
        h = h * (scale.view(h.shape[0], h.shape[1], 1, 1) + 1) + shift.view(h.shape[0], h.shape[1], 1, 1)
        
        return h + self.res_conv(x)


class SR3_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, inner_channel=64, 
                 channel_mults=[1, 2, 4, 4, 8, 8], res_blocks=3, time_emb_dim=256):
        super().__init__()

        self.time_embedding = SinusoidalPosEmb(time_emb_dim)
        
        # 1. Time Embedding Network (for the DDPM process)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # 2. LR Conditioning (Upsampling LR image to HR size)
        # The LR image is upsampled and concatenated to the noisy HR image
        self.lr_conditioner = nn.Sequential(
            nn.Conv2d(in_channels * 2, inner_channel, kernel_size=3, padding=1),
            # This convolution handles the concatenated LR image (upsampled) + Noisy HR image (after concatenation)
        )
        
        # The true input channels = noisy_HR (3) + conditioned_LR (3)
        input_c = in_channels * 2 
        
        # 3. U-Net Encoder/Decoder (using the channel_mults for structure)
        
        # --- Encoder (Downsampling) ---
        # The number of downsampling steps is determined by len(channel_mults)
        # The x8 factor is inherent in the U-Net structure required to reach the target resolution
        
        # 4. Final Output Layer
        self.final_conv = nn.Conv2d(inner_channel * channel_mults[0], out_channels, kernel_size=3, padding=1)

    def forward(self, noisy_hr_img, lr_img, t):
        # 1. Time Embedding
        t_emb = self.time_embedding(t)
        time_emb = self.time_mlp(t_emb)
        
        # 2. Condition on LR Image
        # To condition on LR, we must upsample the LR image to the HR size
        # This is typically done via simple interpolation (bicubic) and concatenated with the noisy HR image
        
        # Assumes lr_img is (B, C, H/8, W/8) and noisy_hr_img is (B, C, H, W)
        hr_size = noisy_hr_img.shape[-2:]
        upsampled_lr = F.interpolate(lr_img, size=hr_size, mode='bicubic', align_corners=False)
        
        # Concatenate the condition (LR) and the noisy signal (HR)
        x = torch.cat([noisy_hr_img, upsampled_lr], dim=1) 
        
        # 3. Initial Conv to adjust channels
        x = self.lr_conditioner(x)
        
        # --- U-Net forward pass (omitted for brevity) ---
        # The actual forward path involves:
        # 1. Downsampling blocks (ResBlock + AvgPool)
        # 2. Bottleneck blocks (ResBlock + Attention)
        # 3. Upsampling blocks (ResBlock + Upsample/ConvTranspose) with skip connections
        
        # 4. Final Output (Predicts the noise, or the cleaned image)
        return self.final_conv(x)

class Diffusion(nn.Module):
    """
    Implements a DDPM-style (Denoising Diffusion Probabilistic Model)
    forward diffusion process, reverse denoising process, and sampling loop.

    Attributes
    ----------
    n_times : int
        Number of diffusion steps.

    img_H, img_W, img_C : int
        Image height, width, and channel count.

    model : nn.Module
        Noise prediction model.

    sqrt_betas : torch.Tensor (T,)
        Square roots of betas for all diffusion steps.

    alphas : torch.Tensor (T,)
        alpha = 1 - beta for each timestep.

    sqrt_alphas : torch.Tensor (T,)
        square root alpha values for all timesteps.

    sqrt_alpha_bars : torch.Tensor (T,)
        Precomputed cumulative product of sqrt_alphas

    sqrt_one_minus_alpha_bars : torch.Tensor (T,)
        used in forward diffusion.

    device : str
        Device used for computation.
    """

    def __init__(self, model, image_resolution=[640, 640, 3], n_times=1000, device='cuda'):
        """
        Initialises the diffusion model

        Parameters
        ----------
        model : nn.Module
            A neural network that predicts the noise `epsilon` given a noisy
            image x_t and a timestep t.

        image_resolution : list or tuple of length 3
            The (H, W, C) resolution of input images.

        n_times : int
            Number of diffusion time steps T. Defaults to 1000.

        beta_minmax : list or tuple of two floats
            Minimum and maximum beta values used to construct a linear variance
            schedule.

        device : str
            Device on which tensors should be allocated
            (e.g., 'cuda' or 'cpu').
        """
        super(Diffusion, self).__init__()
        self.n_times = n_times
        self.img_H, self.img_W, self.img_C = image_resolution
        self.model = model
        # Define linear variance schedule (betas)
        betas = self.cosine_schedule(n_times).to(device)

        self.sqrt_betas = torch.sqrt(betas)
        # Define alphas for forward diffusion process
        self.alphas = 1 - betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        alpha_bars_full = torch.cat([torch.tensor([1.0]).to(device), alpha_bars]) 
        alpha_bars_prev = alpha_bars_full[:-1]

        # Calculate posterior variance (tilde_beta)
        self.posterior_variance = (1 - alpha_bars_prev) / (1 - alpha_bars) * betas
        self.sqrt_posterior_variance = torch.sqrt(self.posterior_variance)

        self.sqrt_one_minus_alpha_bars = torch.sqrt(1-alpha_bars)
        self.device = device

    def cosine_schedule(self, num_timesteps, s=0.008):
        def f(t):
            return torch.cos((t / num_timesteps + s) / (1 + s) * 0.5 * torch.pi) ** 2

        x = torch.linspace(0, num_timesteps, num_timesteps + 1)
        alphas_cumprod = f(x) / f(torch.tensor([0]))
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = torch.clip(betas, 0.0001, 0.02)
        return betas

    def extract(self, a, t, x_shape):
        """
        Extract timestep-dependent constants from a 1D tensor and reshape
        them to broadcast over an image batch.

        Parameters
        ----------
        a : torch.Tensor (T,)
            Schedule values (e.g., sqrt_alpha_bars).

        t : torch.Tensor (B,)
            Batch of timesteps, each in range [0, T).

        x_shape : tuple
            Shape of the target tensor (B, C, H, W) for broadcasting.

        Returns
        -------
        torch.Tensor
            Extracted values reshaped to (B, 1, 1, 1).
        """
        # Extract the specific values for the batch of time-steps `t`
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def scale_to_minus_one_to_one(self, x):
        """
        Scale input image from range [0, 1] to [-1, 1].

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Scaled tensor.
        """
        # Scale input `x` from [0, 1] to [-1, 1]
        return x * 2 - 1

    def reverse_scale_to_zero_to_one(self, x):
        """
        Rescale images from [-1, 1] back to [0, 1].

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Rescaled tensor.
        """
        # Scale input `x` from [-1, 1] back to [0, 1]
        return (x + 1) * 0.5

    def make_noisy(self, x_zeros, t):
        """
        Apply forward diffusion to produce x_t given clean image x_0.

        Parameters
        ----------
        x_zeros : torch.Tensor (B, C, H, W)
            Batch of clean input images x_0, scaled to [-1, 1].

        t : torch.Tensor (B,)
            Diffusion timestep for each image.

        Returns
        -------
        noisy_sample : torch.Tensor (B, C, H, W)
            Noised images x_t.

        epsilon : torch.Tensor (B, C, H, W)
            Ground-truth noise used to generate x_t.
        """
        # Perturb `x_0` into `x_t` (forward diffusion process)
        epsilon = torch.randn_like(x_zeros).to(self.device)
        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, t, x_zeros.shape)
        sqrt_one_minus_alpha_bar = self.extract(
            self.sqrt_one_minus_alpha_bars, t, x_zeros.shape)
        # Let's make noisy sample!: i.e., Forward process with fixed variance schedule
        #      i.e., sqrt(alpha_bar_t) * x_zero + sqrt(1-alpha_bar_t) * epsilon
        noisy_sample = x_zeros * sqrt_alpha_bar + epsilon * sqrt_one_minus_alpha_bar
        return noisy_sample.detach(), epsilon

    def forward(self, lr_img, hr_img):
        """
        Perform a full DDPM training step:
            1. Scale images to [-1, 1].
            2. Sample random timestep t.
            3. Generate perturbed images x_t.
            4. Predict noise using the model.

        Parameters
        ----------
        lr_img : torch.Tensor (B, C, H, W)
            Low resolution input images in range [0, 1].
        hr_img : torch.Tensor (B, C, H, W)
            High resolution images in range [0, 1].

        Returns
        -------
        perturbed_images : torch.Tensor
            Noisy images x_t.

        epsilon : torch.Tensor
            True noise used to create x_t.

        pred_epsilon : torch.Tensor
            Model prediction of noise for loss computation.
        """

        # 1. Scale images to [-1, 1]
        hr_img_scaled = self.scale_to_minus_one_to_one(hr_img)
        lr_img_scaled = self.scale_to_minus_one_to_one(lr_img)
        
        B, _, _, _ = hr_img_scaled.shape
        
        # 2. Randomly select a diffusion time-step `t`
        t = torch.randint(low=0, high=self.n_times, size=(B,)).long().to(self.device)
        
        # 3. Forward diffusion: perturb HR image (x_0)
        perturbed_images, epsilon = self.make_noisy(hr_img_scaled, t)
        
        # 4. Predict the noise (epsilon) using the CONDITIONAL model
        pred_epsilon = self.model(perturbed_images, lr_img_scaled, t)
        
        return perturbed_images, epsilon, pred_epsilon

    def denoise_at_t(self, x_t, lr_img_scaled, timestep, t):
        """
        Perform one reverse denoising step, computing x_{t-1} from x_t.

        Parameters
        ----------
        x_t : torch.Tensor (B, C, H, W)
            Current noisy image at timestep t.

        timestep : torch.Tensor (B,)
            Tensor containing the timestep value t for each sample.

        t : int
            The integer timestep (used to decide whether noise z is added).

        Returns
        -------
        torch.Tensor
            Estimated x_{t-1}, clamped to [-1, 1].
        """
        B, _, _, _ = x_t.shape
        # Generate random noise `z` for sampling, except for the final step (`t=0`)
        if t > 1:
            z = torch.randn_like(x_t).to(self.device)
        else:
            z = torch.zeros_like(x_t).to(self.device)
        # at inference, we use predicted noise(epsilon) to restore perturbed data sample.
        # Use the model to predict noise (`epsilon_pred`) given `x_t` at `timestep`
        epsilon_pred = self.model(x_t, lr_img_scaled, timestep)
        alpha = self.extract(self.alphas, timestep, x_t.shape)
        sqrt_alpha = self.extract(self.sqrt_alphas, timestep, x_t.shape)
        sqrt_one_minus_alpha_bar = self.extract(
            self.sqrt_one_minus_alpha_bars, timestep, x_t.shape)
        sqrt_posterior_variance = self.extract(self.sqrt_posterior_variance, timestep, x_t.shape)
        # denoise at time t, denoise `x_t` to estimate `x_{t-1}`
        x_t_minus_1 = 1 / sqrt_alpha * \
            (x_t - (1-alpha)/sqrt_one_minus_alpha_bar*epsilon_pred) + sqrt_posterior_variance*z
        return x_t_minus_1.clamp(-1., 1)

    def sample(self, lr_img):
        """
        Generate new images by denoising pure Gaussian noise.

        Parameters
        ----------
        lr_img : torch.Tensor (B, C, H, W)
            The low resolition image to be sampled.

        Returns
        -------
        torch.Tensor (N, C, H, W)
            Fully denoised images in range [0, 1].
        """
        N = lr_img.shape[0]
        # 1. Prepare LR image condition
        lr_img_scaled = self.scale_to_minus_one_to_one(lr_img)
        
        # 2. Start from random noise vector x_T at the HR size
        _, C, H, W = lr_img_scaled.shape
        x_t = torch.randn((N, self.img_C, H * 8, W * 8)).to(self.device) # Assuming x8 upscale
        
        # 3. Autoregressively denoise
        for t in range(self.n_times - 1, -1, -1):
            timestep = torch.tensor([t]).repeat_interleave(N, dim=0).long().to(self.device)
            
            # Pass the fixed LR condition to the denoise step
            x_t = self.denoise_at_t(x_t, lr_img_scaled, timestep, t) # <--- PASS LR_IMG_SCALED
            
        # Convert the final result x_0 back to [0, 1] range
        x_0 = self.reverse_scale_to_zero_to_one(x_t)
        return x_0

    def speedSample(self, lr_img):
        """
        Generate new images by denoising pure Gaussian noise, smaller subsampling so completes faster

        Parameters
        ----------
        lr_img : torch.Tensor (B, C, H, W)
            The low resolition image to be sampled.

        Returns
        -------
        torch.Tensor (N, C, H, W)
            Fully denoised images in range [0, 1].
        """
        N = lr_img.shape[0]
        S = 100
        # 1. Prepare LR image condition
        lr_img_scaled = self.scale_to_minus_one_to_one(lr_img)
        
        # 2. Start from random noise vector x_T at the HR size
        _, C, H, W = lr_img_scaled.shape
        x_t = torch.randn((N, self.img_C, H * 8, W * 8)).to(self.device) # Assuming x8 upscale
        
        # 3. Autoregressively denoise
        for t in range(self.n_times - 1, 0, -self.n_times // S):
            timestep = torch.tensor([t]).repeat_interleave(N, dim=0).long().to(self.device)
            
            # Pass the fixed LR condition to the denoise step
            x_t = self.denoise_at_t(x_t, lr_img_scaled, timestep, t) # <--- PASS LR_IMG_SCALED
            
        # Convert the final result x_0 back to [0, 1] range
        x_0 = self.reverse_scale_to_zero_to_one(x_t)
        return x_0