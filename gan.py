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
    def __init__(self, in_ch=3, out_ch=3, num_feat=64, num_blocks=6, gc=32, scale=8):
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
    def __init__(self, in_channels=3, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()

        def vgg_sn_block(in_feat, out_feat, stride=2, use_sn=True):
            conv = nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=stride, padding=1, bias=True)
            
            # Apply Spectral Normalization
            if use_sn:
                layers = [spectral_norm(conv)]
            else:
                layers = [conv]
                
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []

        layers.extend(vgg_sn_block(in_channels, ndf, stride=2, use_sn=True))

        # Downsampling 
        nf_mult = 1
        nf_mult_prev = 1
        
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            
            layers.extend(vgg_sn_block(ndf * nf_mult_prev, ndf * nf_mult, stride=2, use_sn=True))

        # Refinement Layer
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        
        # Conv with SN
        conv_intermediate = nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=3, stride=1, padding=1, bias=True)
        layers.append(spectral_norm(conv_intermediate))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Output Layer
        layers.append(nn.Conv2d(ndf * nf_mult, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)