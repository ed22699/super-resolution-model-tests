import torch
import torch.nn as nn
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


class ConvBlock(nn.Conv2d):
    """
    A configurable 2D convolutional block that optionally applies
    activation, dropout, and group normalization after the convolution.

    Inherits directly from `nn.Conv2d`, meaning all convolution
    parameters behave exactly the same as PyTorch's Conv2d module.

    Attributes
    ----------
    activation_fn : nn.Module or None
        Optional activation function applied after the convolution.
        Currently uses `nn.SiLU` if enabled.
    group_norm : nn.Module or None
        Optional normalization layer. Uses `nn.GroupNorm` when enabled.
    drop_rate : float
        The dropout probability. (Dropout layer is applied only if > 0.)
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 activation_fn=None, drop_rate=0., stride=1,
                 padding='same', dilation=1, groups=1,
                 bias=True, gn=False, gn_groups=8):
        """
        Initialize a convolutional block with optional activation,
        dropout, and group normalization.

        Parameters
        ----------
        in_channels : int
            Number of channels in the input tensor.
        out_channels : int
            Number of channels produced by the convolution.
        kernel_size : int
            Size of the convolutional kernel.
        activation_fn : bool, optional
            If True, applies a SiLU activation after the convolution.
        drop_rate : float, optional
            Dropout probability. If 0, no dropout layer is applied.
        stride : int, optional
            Convolution stride. Default is 1.
        padding : str or int, optional
            If 'same', padding is automatically calculated to preserve
            spatial dimensions (accounting for dilation). Otherwise,
            can pass an integer for explicit padding.
        dilation : int, optional
            Spacing between kernel elements. Default is 1.
        groups : int, optional
            Number of blocked connections from input to output channels.
        bias : bool, optional
            If True, adds a learnable bias to the convolution.
        gn : bool, optional
            If True, enables Group Normalization after convolution.
        gn_groups : int, optional
            Number of groups to use for GroupNorm when `gn=True`.
        """

        if padding == 'same':
            padding = kernel_size // 2 * dilation

        super(ConvBlock, self).__init__(in_channels, out_channels, kernel_size,
                                        stride=stride, padding=padding,
                                        dilation=dilation,
                                        groups=groups, bias=bias)

        self.activation_fn = nn.SiLU() if activation_fn else None
        self.group_norm = nn.GroupNorm(gn_groups, out_channels) if gn else None

    def forward(self, x, time_embedding=None, residual=False):
        """
        forward pass through the convolutional block

        Parameters
        ----------
        x : torch.Tensor
            Batch of feature maps (batch_size, channels, height, width)
        time_embedding : torch.Tensor, optional
            Tensor containing timestep information
        residual : bool
            If true adds time_embedding to x

        Returns
        -------
        torch.Tensor
            Output tensor after convolution, normalization,
            and activation.
        """
        if residual:
            x = x + time_embedding
            y = x
            x = super(ConvBlock, self).forward(x)
            y = y + x
        else:
            y = super(ConvBlock, self).forward(x)
        y = self.group_norm(y) if self.group_norm is not None else y
        y = self.activation_fn(y) if self.activation_fn is not None else y
        return y


class Denoiser(nn.Module):
    """
    A model that removes noise from images at different steps of the diffusion
    process.

    Attributes
    ----------
    time_embedding : nn.Module
        Module that converts a scalar diffusion timestep into
        a sinusoidal embedding vector of dimension
        `diffusion_time_embedding_dim`.

    in_project : ConvBlock
        Initial convolutional projection that maps the input
        image (img_C channels) into the hidden feature space.

    time_project : nn.Sequential
        Two-layer convolutional projection that transforms the
        timestep embedding and produces features

    convs : nn.ModuleList
        A sequence of ConvBlocks that form the main body of the
        denoising network.

    out_project : ConvBlock
        Final convolutional projection that maps the hidden
        features back to an image with `img_C` channels.
    """

    def __init__(self, image_resolution, hidden_dims=[256, 256], diffusion_time_embedding_dim=256, n_times=1000):
        """
        Initialize the Denoiser network.

        Parameters
        ----------
        image_resolution : tuple
            A tuple (H, W, C) giving the input image resolution.

        hidden_dims : [int], optional
            Channel dimensions for each internal ConvBlock in the
            network. Defines model capacity. Default [256, 256].

        diffusion_time_embedding_dim : int, optional
            Dimensionality of the sinusoidal timestep embedding.
            Default is 256.

        n_times : int, optional
            Total number of diffusion steps. Default is 1000.
        """
        super(Denoiser, self).__init__()
        _, _, img_C = image_resolution
        self.time_embedding = SinusoidalPosEmb(diffusion_time_embedding_dim)
        self.in_project = ConvBlock(img_C, hidden_dims[0], kernel_size=7)
        self.time_project = nn.Sequential(
            ConvBlock(diffusion_time_embedding_dim,
                      hidden_dims[0], kernel_size=1, activation_fn=True),
            ConvBlock(hidden_dims[0], hidden_dims[0], kernel_size=1))
        self.convs = nn.ModuleList([ConvBlock(
            in_channels=hidden_dims[0], out_channels=hidden_dims[0], kernel_size=3)])
        for idx in range(1, len(hidden_dims)):
            self.convs.append(ConvBlock(hidden_dims[idx-1], hidden_dims[idx], kernel_size=3, dilation=3**((idx-1)//2),
                                        activation_fn=True, gn=True, gn_groups=8))
        self.out_project = ConvBlock(
            hidden_dims[-1], out_channels=img_C, kernel_size=3)

    def forward(self, perturbed_x, diffusion_timestep):
        """
        Forward pass through the denoiser network.

        Parameters
        ----------
        perturbed_x : torch.Tensor
            A 4D tensor of shape (B, C, H, W) representing the noisy
            input image batch.

        diffusion_timestep : torch.Tensor
            A 1D tensor of shape (B,) giving the diffusion timestep
            for each sample in the batch, converted into a time embedding.

        Returns
        -------
        torch.Tensor
            A 4D tensor of shape (B, C, H, W) representing the predicted noise.
        """
        y = perturbed_x
        diffusion_embedding = self.time_embedding(diffusion_timestep)
        diffusion_embedding = self.time_project(
            diffusion_embedding.unsqueeze(-1).unsqueeze(-2))
        y = self.in_project(y)
        for i in range(len(self.convs)):
            y = self.convs[i](y, diffusion_embedding, residual=True)
        y = self.out_project(y)
        return y
        y = perturbed_x
        diffusion_embedding = self.time_embedding(diffusion_timestep)
        diffusion_embedding = self.time_project(
            diffusion_embedding.unsqueeze(-1).unsqueeze(-2))
        y = self.in_project(y)
        for i in range(len(self.convs)):
            y = self.convs[i](y, diffusion_embedding, residual=True)
        y = self.out_project(y)
        return y


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

    def __init__(self, model, image_resolution=[28, 28, 1], n_times=1000, beta_minmax=[1e-4, 2e-2], device='cuda'):
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
        beta_1, beta_T = beta_minmax
        betas = torch.linspace(start=beta_1, end=beta_T, steps=n_times).to(
            device)  # follows DDPM paper: cosine function instead of linear?
        self.sqrt_betas = torch.sqrt(betas)
        # Define alphas for forward diffusion process
        self.alphas = 1 - betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1-alpha_bars)
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.device = device

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

    def forward(self, x_zeros):
        """
        Perform a full DDPM training step:
            1. Scale images to [-1, 1].
            2. Sample random timestep t.
            3. Generate perturbed images x_t.
            4. Predict noise using the model.

        Parameters
        ----------
        x_zeros : torch.Tensor (B, C, H, W)
            Clean input images in range [0, 1].

        Returns
        -------
        perturbed_images : torch.Tensor
            Noisy images x_t.

        epsilon : torch.Tensor
            True noise used to create x_t.

        pred_epsilon : torch.Tensor
            Model prediction of noise for loss computation.
        """
        x_zeros = self.scale_to_minus_one_to_one(x_zeros)
        B, _, _, _ = x_zeros.shape
        # 1. Randomly select a diffusion time-step `t`
        t = torch.randint(low=0, high=self.n_times,
                          size=(B,)).long().to(self.device)
        # 2. Forward diffusion: perturb `x_zeros` using the fixed variance schedule
        perturbed_images, epsilon = self.make_noisy(x_zeros, t)
        # 3. Predict the noise (`epsilon`) given the perturbed image at time-step `t`
        pred_epsilon = self.model(perturbed_images, t)
        return perturbed_images, epsilon, pred_epsilon

    def denoise_at_t(self, x_t, timestep, t):
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
        epsilon_pred = self.model(x_t, timestep)
        alpha = self.extract(self.alphas, timestep, x_t.shape)
        sqrt_alpha = self.extract(self.sqrt_alphas, timestep, x_t.shape)
        sqrt_one_minus_alpha_bar = self.extract(
            self.sqrt_one_minus_alpha_bars, timestep, x_t.shape)
        sqrt_beta = self.extract(self.sqrt_betas, timestep, x_t.shape)
        # denoise at time t, denoise `x_t` to estimate `x_{t-1}`
        x_t_minus_1 = 1 / sqrt_alpha * \
            (x_t - (1-alpha)/sqrt_one_minus_alpha_bar*epsilon_pred) + sqrt_beta*z
        return x_t_minus_1.clamp(-1., 1)

    def sample(self, N):
        """
        Generate new images by denoising pure Gaussian noise.

        Parameters
        ----------
        N : int
            Number of images to sample.

        Returns
        -------
        torch.Tensor (N, C, H, W)
            Fully denoised images in range [0, 1].
        """
        # Start from random noise vector `x_T`, x_0 (for simplicity, x_T declared as x_t instead of x_T)
        x_t = torch.randn((N, self.img_C, self.img_H,
                          self.img_W)).to(self.device)
        # Autoregressively denoise from `x_T` to `x_0`
        #     i.e., generate image from noise, x_T
        for t in range(self.n_times-1, -1, -1):
            timestep = torch.tensor([t]).repeat_interleave(
                N, dim=0).long().to(self.device)
            x_t = self.denoise_at_t(x_t, timestep, t)
        # Convert the final result `x_0` back to [0, 1] range
        x_0 = self.reverse_scale_to_zero_to_one(x_t)
        return x_0
