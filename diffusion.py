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
        x = x.float() 
        half_dim = self.dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, device=device).float() / half_dim))
        emb = x[:, None] * inv_freq[None, :] 
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x: (B,) of integers/longs or floats
        device = x.device
        x = x.float()  # IMPORTANT: ensure float
        half_dim = self.dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, device=device).float() / half_dim))
        # shape: (half_dim,)
        emb = x[:, None] * inv_freq[None, :]  # (B, half_dim)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb  # (B, dim)


class ResBlock(nn.Module):
    # A simplified Residual Block, often adapted from BigGAN or DDPM
    def __init__(self, in_c, out_c, time_emb_dim, norm_groups=8):
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

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Kernel 3, Stride 2, Padding 1 = Halves dimension (H/2, W/2)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))

# -----------------------------
# SR3 U-Net
# -----------------------------
class SR3UNet(nn.Module):
    def __init__(self, in_channels=3, cond_channels=3, base_channels=64, channel_mults=[1,2,4, 8], time_dim=256):
        super().__init__()
        
        self.time_dim = time_dim
        
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim*4),
            nn.SiLU(),
            nn.Linear(time_dim*4, time_dim)
        )

        # Input convolution
        self.init_conv = nn.Conv2d(in_channels + cond_channels, base_channels, 3, padding=1)
        
        # -----------------------------
        # Encoder (Down)
        # -----------------------------
        self.downs = nn.ModuleList()
        in_ch = base_channels
        
        # We need to store output channels to know what size skips will be
        self.skip_channels = [] 
        
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            
            # ResBlock
            self.downs.append(ResBlock(in_ch, out_ch, time_dim))
            self.skip_channels.append(out_ch) # Save for decoder
            
            # Downsample (except last)
            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(out_ch))
                # Downsample doesn't change channels
                
            in_ch = out_ch

        # -----------------------------
        # Mid Block
        # -----------------------------
        self.mid_block1 = ResBlock(in_ch, in_ch, time_dim)
        self.mid_block2 = ResBlock(in_ch, in_ch, time_dim)
        
        # -----------------------------
        # Decoder (Up)
        # -----------------------------
        self.ups = nn.ModuleList()
        
        # Iterate in reverse: [4, 2, 1]
        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            
            # Retrieve the skip channel size corresponding to this level
            # We pop from the end of our recorded list
            skip_ch = self.skip_channels.pop()
            
            # ResBlock Input = Current Up Features + Skip Features
            self.ups.append(ResBlock(in_ch + skip_ch, out_ch, time_dim))
            
            # Upsample (except last)
            if i != len(channel_mults) - 1:
                self.ups.append(Upsample(out_ch))
                
            in_ch = out_ch

        # Final output
        self.final_conv = nn.Conv2d(in_ch, 3, 1)

    def forward(self, x_noisy, cond, t):
        cond_up = F.interpolate(cond, size=x_noisy.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x_noisy, cond_up], dim=1)
        
        t_emb = self.time_mlp(t)
        
        # Initial Conv
        h = self.init_conv(x)
        
        # Encoder
        skips = []
        for layer in self.downs:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
                skips.append(h) # Save skip
            else:
                h = layer(h) # Downsample
        
        # Middle
        h = self.mid_block1(h, t_emb)
        h = F.silu(h)
        h = self.mid_block2(h, t_emb)

        # Decoder
        for layer in self.ups:
            if isinstance(layer, ResBlock):
                # We expect the list of skips to align with our layers
                skip = skips.pop() 
                
                # Check shapes to debug (Optional)
                # if h.shape[2:] != skip.shape[2:]:
                #     h = F.interpolate(h, size=skip.shape[2:], mode='nearest')
                
                h = torch.cat([h, skip], dim=1)
                h = layer(h, t_emb)
            else:
                h = layer(h) # Upsample
        
        return self.final_conv(h)

class Diffusion:
    def __init__(self, model, timesteps=1000, device="cuda"):
        self.model = model
        self.timesteps = timesteps
        self.device = device
        
        # Linear beta schedule
        self.betas = self.cosine_schedule(timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def cosine_schedule(self, num_timesteps, s=0.008):
        def f(t):
            return torch.cos((t / num_timesteps + s) / (1 + s) * 0.5 * torch.pi) ** 2

        x = torch.linspace(0, num_timesteps, num_timesteps + 1)
        alphas_cumprod = f(x) / f(torch.tensor([0]))
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = torch.clip(betas, 0.0001, 0.02)
        return betas

    # q(x_t | x_0)  (forward) = add noise
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha = self.alpha_cumprod[t].sqrt().view(-1,1,1,1)
        sqrt_one_minus_alpha = (1 - self.alpha_cumprod[t]).sqrt().view(-1,1,1,1)
        
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise, noise

    # Training loss
    def p_losses(self, x0, cond, t):
        x_t, noise = self.q_sample(x0, t)
        predicted_noise = self.model(x_t, cond, t)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, cond, shape):
        x = torch.randn(shape).to(self.device)
        
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            predicted_noise = self.model(x, cond, t)
            
            beta = self.betas[i]
            alpha = self.alphas[i]
            alpha_cumprod = self.alpha_cumprod[i]
            
            coef1 = 1 / alpha.sqrt()
            coef2 = beta / (1 - alpha_cumprod).sqrt()
            
            # Mean calculation
            mean = coef1 * (x - coef2 * predicted_noise)
            mean.clamp_(-1., 1.)

            if i > 0:
                noise = torch.randn_like(x)
                
                # --- SIMPLIFIED VARIANCE (Fixes Grain) ---
                # Old way: complex formula using alpha_prev
                # New way: standard DDPM sigma
                sigma = beta.sqrt()
                
                x = mean + sigma * noise
            else:
                x = mean
                
        # Return [-1, 1] output
        return x.clamp(-1., 1.)
