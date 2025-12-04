import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=64, patch_size=1):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x)


class PatchUnEmbed(nn.Module):
    def __init__(self, out_ch=3, embed_dim=64):
        super().__init__()
        self.conv = nn.Conv2d(embed_dim, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


# --- FIXED WINDOW FUNCTIONS ---
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
    Returns:
        windows: (num_windows*B, window_size*window_size, C)
    """
    B, H, W, C = x.shape
    # 1. Reshape to separate windows (Safe for non-contiguous memory)
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    
    # 2. Permute and Flatten
    # Output becomes: (Batch*NumWindows, WindowArea, Channels)
    # e.g., (-1, 64, 64) for 8x8 window
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size*window_size, C)
    Returns:
        x: (B, H, W, C)
    """
    # Calculate Batch size
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    
    # Reshape back
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        
        # Determine head dimension
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: (B_, N, C)
        B_, N, C = x.shape
        
        # Reshape for multi-head attention
        # (B_, N, 3, Heads, HeadDim)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * (self.head_dim ** -0.5)
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class ShallowCNN(nn.Module):
    def __init__(self, in_ch=3, dim=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, dim, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(dim, dim, 3, padding=1)
        )

    def forward(self, x):
        return self.layers(x)

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # 1. PADDING (Prevents Crashes on Odd Sizes)
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        
        _, Hp, Wp, _ = x.shape

        # 2. Partition
        x_windows = window_partition(x, self.window_size)
        
        # 3. Attention
        attn_windows = self.attn(x_windows) 

        # 4. Reverse
        x = window_reverse(attn_windows, self.window_size, Hp, Wp) 

        # 5. Remove Padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :]

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class RSTB(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(dim, num_heads, window_size) for _ in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x, H, W):
        shortcut = x
        for blk in self.blocks:
            x = blk(x, H, W)

        B, L, C = x.shape
        x_img = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x_img = self.conv(x_img)
        x = x_img.flatten(2).transpose(1, 2)
        return shortcut + x


class Upsample(nn.Module):
    def __init__(self, scale, dim):
        super().__init__()
        m = []
        if scale == 2:
            m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.PixelShuffle(3))
        elif scale == 4:
            m.append(nn.Conv2d(dim, dim * 4, 3, 1, 1))
            m.append(nn.PixelShuffle(2))
            m.append(nn.Conv2d(dim, dim * 4, 3, 1, 1))
            m.append(nn.PixelShuffle(2))
        elif scale == 8:
            # 8x Upscaling Logic
            m.append(nn.Conv2d(dim, dim * 4, 3, 1, 1))
            m.append(nn.PixelShuffle(2))
            m.append(nn.Conv2d(dim, dim * 4, 3, 1, 1))
            m.append(nn.PixelShuffle(2))
            m.append(nn.Conv2d(dim, dim * 4, 3, 1, 1))
            m.append(nn.PixelShuffle(2))
            
        self.up = nn.Sequential(*m)

    def forward(self, x):
        return self.up(x)


class SwinIR(nn.Module):
    def __init__(
        self,
        in_ch=3,
        embed_dim=64, # Default 64
        depths=[6, 6, 6, 6],
        num_heads=[4, 4, 4, 4], # Default 4 heads (Divides 64 and 128 evenly)
        window_size=8,
        upscale=8,
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.upscale = upscale

        # shallow feature extractor
        self.shallow_cnn = ShallowCNN(in_ch, embed_dim)
        self.patch_embed = PatchEmbed(in_ch=embed_dim, embed_dim=embed_dim, patch_size=1)
        self.patch_unembed = PatchUnEmbed(in_ch, embed_dim)

        # deep Swin Transformer layers (RSTBs)
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim, 
                depth=depths[i], 
                num_heads=num_heads[i], 
                window_size=window_size
            )
            self.layers.append(layer)

        # upsampler
        self.upsample = Upsample(upscale, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        # shallow features
        x = self.shallow_cnn(x)
        feat = self.patch_embed(x)
        feat = feat.flatten(2).transpose(1, 2) # (B, L, C)

        # deep features
        for layer in self.layers:
            feat = layer(feat, H, W)

        # unembed
        feat = feat.transpose(1, 2).view(B, -1, H, W)

        # upsample
        out = self.upsample(feat)

        # reconstruction
        out = self.patch_unembed(out)
        return out