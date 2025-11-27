import torch
from torch import nn
import functools
import torch.nn.functional as F
import torch.nn.init as init
from torchvision.models.vgg import vgg19
from torch.nn.utils import spectral_norm

class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=36, requires_grad=False):
        super().__init__()
        
        # Load pre-trained VGG-19
        vgg_pretrained_features = vgg19(weights='VGG19_Weights.IMAGENET1K_V1').features
        
        # stop at layer 36
        self.slice = nn.Sequential()
        for x in range(feature_layer + 1):
            self.slice.add_module(str(x), vgg_pretrained_features[x])

        # Freeze the weights
        if not requires_grad:
            for param in self.slice.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.slice(x)


def initialise_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

# -----------------------
# Residual Dense Block
# -----------------------
class DenseBlock(nn.Module):
    def __init__(self, channels=64, growth=32, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channels + growth, growth, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channels + 2 * growth, growth, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channels + 3 * growth, growth, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channels + 4 * growth, channels, 3, 1, 1, bias=bias)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        # initialisation
        initialise_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        f1 = self.lrelu(self.conv1(x))
        f2 = self.lrelu(self.conv2(torch.cat((x, f1), 1)))
        f3 = self.lrelu(self.conv3(torch.cat((x, f1, f2), 1)))
        f4 = self.lrelu(self.conv4(torch.cat((x, f1, f2, f3), 1)))
        f5 = self.conv5(torch.cat((x, f1, f2, f3, f4), 1))

        return f5 * 0.2 + x


# -----------------------
# RRDB Block (Residual in Residual Dense Block)
# -----------------------
class RRDB(nn.Module):
    def __init__(self, channels, growth=32):
        super().__init__()
        self.db1 = DenseBlock(channels, growth)
        self.db2 = DenseBlock(channels, growth)
        self.db3 = DenseBlock(channels, growth)

    def forward(self, x):
        out = self.db1(x)
        out = self.db2(out)
        out = self.db3(out)
        return x + 0.2 * out


# -----------------------
# BSRGAN / ESRGAN Generator RRDBNet
# -----------------------
class Generator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, num_feat=64, num_blocks=8, gc=32, scale=8):
        super().__init__()
        RRDB_block_f = functools.partial(RRDB, channels=num_feat, growth=gc)
        self.scale = scale

        # First layer
        self.conv_first = nn.Conv2d(in_ch, num_feat, 3, 1, 1, bias=True)

        # RRDB trunk
        self.RRDB_trunk = make_layer(RRDB_block_f, num_blocks)
        self.trunk_conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)

        # Upsampling modules
        self.upconv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        if self.scale>=4:
            self.upconv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)

        if self.scale == 8:
            self.upconv3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)

        # Initialize the upsampling layers
        if hasattr(self, 'upconv2'):
            initialise_weights(self.upconv2, 1.0)
        if hasattr(self, 'upconv3'):
            initialise_weights(self.upconv3, 1.0)

        self.HRconv = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)

        initialise_weights([self.conv_first, self.trunk_conv, self.HRconv, self.upconv1], 1.0) 

        # Final conv
        self.conv_last = nn.Conv2d(num_feat, out_ch, 3, 1, 1, bias=True)

        nn.init.normal_(self.conv_last.weight, mean=0, std=0.001)
        nn.init.constant_(self.conv_last.bias, 0.5)
        # initialise_weights(self.conv_last, 1.0) 

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        if self.scale>=4:
            fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))

        if self.scale == 8:
            fea = self.lrelu(self.upconv3(F.interpolate(fea, scale_factor=2, mode='nearest')))


        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out


class Discriminator(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64):
        super(Discriminator, self).__init__()
        
        # Helper to create spectral norm convolution
        def sn_conv(in_channels, out_channels, kernel_size, stride, padding, bias=True):
            k = 4 if stride == 2 else 3 
            p = 1 if stride == 2 else 1
            return spectral_norm(nn.Conv2d(in_channels, out_channels, k, stride, p, bias=bias))

        # 1. Downsampling (Encoder - 3 levels + initial conv)
        nf = num_feat # 64
        
        # Encoder convolutions
        self.conv0 = sn_conv(num_in_ch, nf, 3, 1, 1)        # 3 -> 64
        self.conv1 = sn_conv(nf, nf * 2, 4, 2, 1, bias=False)       # 64 -> 128 (H/2)
        self.conv2 = sn_conv(nf * 2, nf * 4, 4, 2, 1, bias=False)   # 128 -> 256 (H/4)
        self.conv3 = sn_conv(nf * 4, nf * 8, 4, 2, 1, bias=False)   # 256 -> 512 (H/8)
        
        # 2. Upsampling (Decoder - 3 levels)
        # UPDATE THESE LINES TO MATCH CONCATENATION SIZES
        
        # conv4: (512 from x3_up) + (256 from x2) = 768 channels input
        self.conv4 = sn_conv(nf * 8 + nf * 4, nf * 4, 3, 1, 1) # 768 -> 256
        
        # conv5: (256 from x4_up) + (128 from x1) = 384 channels input
        self.conv5 = sn_conv(nf * 4 + nf * 2, nf * 2, 3, 1, 1) # 384 -> 128
        
        # conv6: (128 from x5_up) + (64 from x0) = 192 channels input
        self.conv6 = sn_conv(nf * 2 + nf, nf, 3, 1, 1) # 192 -> 64

        # 3. Final Output
        self.conv7 = sn_conv(nf, nf, 3, 1, 1)
        self.conv_last = sn_conv(nf, 1, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # -- Encoder --
        x0 = self.lrelu(self.conv0(x)) # Skip 0 (64ch)
        x1 = self.lrelu(self.conv1(x0)) # Skip 1 (128ch)
        x2 = self.lrelu(self.conv2(x1)) # Skip 2 (256ch)
        x3 = self.lrelu(self.conv3(x2)) # Deepest features (512ch)

        # -- Decoder (with concatenation-based skip connections) --
        # Up 1: Combine 512ch (upsampled x3) and 256ch (x2)
        x3_up = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        # Concat: (N, 512+256, H/4, W/4)
        x_concat_1 = torch.cat((x3_up, x2), dim=1) 
        x4 = self.lrelu(self.conv4(x_concat_1)) # (N, 256, H/4, W/4)
        
        # Up 2: Combine 256ch (upsampled x4) and 128ch (x1)
        x4_up = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        # Concat: (N, 256+128, H/2, W/2)
        x_concat_2 = torch.cat((x4_up, x1), dim=1)
        x5 = self.lrelu(self.conv5(x_concat_2)) # (N, 128, H/2, W/2)
        
        # Up 3: Combine 128ch (upsampled x5) and 64ch (x0)
        x5_up = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        # Concat: (N, 128+64, H, W)
        x_concat_3 = torch.cat((x5_up, x0), dim=1)
        x6 = self.lrelu(self.conv6(x_concat_3)) # (N, 64, H, W)

        # Final refinement and output map
        out = self.lrelu(self.conv7(x6))
        out = self.conv_last(out)
        
        return out

# class Discriminator(nn.Module):
#     """
#     Defines a U-Net discriminator with Spectral Normalization (SN),
#     deepened to handle 8x upscaled images (e.g., 512x512 input).
#     """
#     def __init__(self, num_in_ch=3, num_feat=64):
#         super(Discriminator, self).__init__()
        
#         # Helper to create spectral norm convolution
#         def sn_conv(in_channels, out_channels, kernel_size, stride, padding, bias=True):
#             return spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))

#         # 1. Downsampling (Encoder - 5 levels for large input)
#         # Input size: H x W
#         self.conv0 = sn_conv(num_in_ch, num_feat, 3, 1, 1) # H x W -> H x W (64)
        
#         self.conv1 = sn_conv(num_feat, num_feat * 2, 4, 2, 1, bias=False)       # H/2 x W/2 (128)
#         self.conv2 = sn_conv(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False)   # H/4 x W/4 (256)
#         self.conv3 = sn_conv(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False)   # H/8 x W/8 (512)
#         self.conv4 = sn_conv(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)   # H/16 x W/16 (512) # NEW
#         self.conv5 = sn_conv(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)   # H/32 x W/32 (512) # NEW

#         # 2. Upsampling (Decoder - 5 levels)
#         self.conv6 = sn_conv(num_feat * 8, num_feat * 8, 3, 1, 1)
#         self.conv7 = sn_conv(num_feat * 8, num_feat * 4, 3, 1, 1)
#         self.conv8 = sn_conv(num_feat * 4, num_feat * 2, 3, 1, 1)
#         self.conv9 = sn_conv(num_feat * 2, num_feat, 3, 1, 1)
#         self.conv10 = sn_conv(num_feat, num_feat, 3, 1, 1)

#         # 3. Final Output
#         self.conv11 = sn_conv(num_feat, num_feat, 3, 1, 1)
#         self.conv_last = sn_conv(num_feat, 1, 3, 1, 1)
        
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

#     def forward(self, x):
#         # -- Encoder --
#         x0 = self.lrelu(self.conv0(x)) # Skip connection 0 (512)
#         x1 = self.lrelu(self.conv1(x0)) # Skip connection 1 (256)
#         x2 = self.lrelu(self.conv2(x1)) # Skip connection 2 (128)
#         x3 = self.lrelu(self.conv3(x2)) # Skip connection 3 (64)
#         x4 = self.lrelu(self.conv4(x3)) # Skip connection 4 (32)
#         x5 = self.lrelu(self.conv5(x4)) # Deepest feature map (16)

#         # -- Decoder (with skip connections) --
#         # Up 1 (to 32)
#         x5_up = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
#         x6 = self.lrelu(self.conv6(x5_up + x4)) 
        
#         # Up 2 (to 64)
#         x6_up = F.interpolate(x6, scale_factor=2, mode='bilinear', align_corners=False)
#         x7 = self.lrelu(self.conv7(x6_up + x3)) 
        
#         # Up 3 (to 128)
#         x7_up = F.interpolate(x7, scale_factor=2, mode='bilinear', align_corners=False)
#         x8 = self.lrelu(self.conv8(x7_up + x2)) 
        
#         # Up 4 (to 256)
#         x8_up = F.interpolate(x8, scale_factor=2, mode='bilinear', align_corners=False)
#         x9 = self.lrelu(self.conv9(x8_up + x1))

#         # Up 5 (to 512)
#         x9_up = F.interpolate(x9, scale_factor=2, mode='bilinear', align_corners=False)
#         x10 = self.lrelu(self.conv10(x9_up + x0))
        
#         # Final refinement and output map
#         out = self.lrelu(self.conv11(x10))
#         out = self.conv_last(out)
        
#         return out
