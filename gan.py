import torch
from torch import nn
import functools
import torch.nn.functional as F
import torch.nn.init as init

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
        f1 = self.lrelu(self.layers[0](x))
        f2 = self.lrelu(self.layers[1](torch.cat((x, f1), 1)))
        f3 = self.lrelu(self.layers[2](torch.cat((x, f1, f2), 1)))
        f4 = self.lrelu(self.layers[3](torch.cat((x, f1, f2, f3), 1)))
        f5 = self.layers[4](torch.cat((x, f1, f2, f3, f4), 1))

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
    def __init__(self, in_ch=3, out_ch=3, num_feat=64, num_blocks=23, gc=32, scale=4):
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
        if self.scale==4:
            self.upconv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)

        # Final conv
        self.conv_last = nn.Conv2d(num_feat, out_ch, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        if self.scale==4:
            fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))

        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out


# class Generator(nn.Module):
#     def __init__(self, input_dim):
#         super(Generator, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 32 * 32)
#         self.br1 = nn.Sequential(
#             nn.BatchNorm1d(1024),
#             nn.ReLU()
#         )
#         self.fc2 = nn.Linear(32 * 32, 128 * 7 * 7)
#         self.br2 = nn.Sequential(
#             nn.BatchNorm1d(128 * 7 * 7),
#             nn.ReLU()
#         )
#         self.conv1 = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#         )
#         self.conv2 = nn.Sequential(
#             nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),  # Final upsampling to 28x28x1
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         x = self.br1(self.fc1(x))
#         x = self.br2(self.fc2(x))
#         # Reshape the tensor for the convolutional layers
#         x = x.reshape(-1, 128, 7, 7)
#         x = self.conv1(x)
#         output = self.conv2(x)
#         return output


# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 32, 5, stride=1),
#             nn.LeakyReLU(0.2)
#         )
#         self.pl1 = nn.MaxPool2d(2, stride=2)
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(32, 64, 5, stride=1),
#             nn.LeakyReLU(0.2)
#         )
#         self.pl2 = nn.MaxPool2d(2, stride=2)
#         self.fc1 = nn.Sequential(
#             nn.Linear(64 * 4 * 4, 1024),
#             nn.LeakyReLU(0.2)
#         )
#         # Output layer: input size = 1024, output size = 1 (probability of being real or fake)
#         self.fc2 = nn.Sequential(
#             nn.Linear(1024, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.pl1(x)
#         x = self.conv2(x)
#         x = self.pl2(x)
#         # Flatten the feature maps into a 1D vector for the fully connected layers
#         x = x.view(x.shape[0], -1)
#         x = self.fc1(x)
#         output = self.fc2(x)
#         return output

class Discriminator(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        nf = 64

        def block(in_feat, out_feat, norm=True):
            layers = [
                nn.Conv2d(in_feat, out_feat, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            if norm:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.Conv2d(out_feat, out_feat, 4, 2, 1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if norm:
                layers.append(nn.BatchNorm2d(out_feat))
            return layers

        self.features = nn.Sequential(
            *block(in_ch, nf, norm=False),        # 64
            *block(nf, nf*2),                     # 128
            *block(nf*2, nf*4),                   # 256
            *block(nf*4, nf*8),                   # 512
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 32 * 32, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

