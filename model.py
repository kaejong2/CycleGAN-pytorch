# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from layer import ConvBlock, ResidualBlock


# class Generator(nn.Module):
#     def __init__(self, input_channels, output_channels, n_residual_blocks=9):
#         super(Generator, self).__init__()

#         # Denote c7s1-k : d7s1
#         self.c7s1_64 = ConvBlock(in_features=input_channels, out_features=64, kernel_size=7, padding=3, padding_mode='reflaction', stride=1, norm='Instance', act="ReLU", drop=False, mode='Conv')

#         # Denode dk : d128, d256
#         self.d128 = ConvBlock(in_features=64, out_features=128, kernel_size=3, padding=1, padding_mode='reflaction', stride=2, norm='Instance', act="ReLU", drop=False, mode='Conv')
#         self.d256 = ConvBlock(in_features=128, out_features=256, kernel_size=3, padding=1, padding_mode='reflaction', stride=2, norm='Instance', act="ReLU", drop=False, mode='Conv')
        
#         # Denote Rk : R256 * 9
#         res = []
#         for _ in range(n_residual_blocks):
#             res += [ResidualBlock(256, 256, kernel_size=3, padding=1, padding_mode='reflaction', stride=1, norm='instance', act='ReLU', drop=False, mode='conv')]
#         self.Rk = nn.Sequential(*res)

#         # Denote uk : u128, u64
#         self.u128 = ConvBlock(in_features=256, out_features=128, kernel_size=4, padding=1, padding_mode='zeros', stride=2, norm='Instance', act="ReLU", drop=False, mode='ConvTranspose')
#         self.u64  = ConvBlock(in_features=128, out_features=64, kernel_size=4, padding=1, padding_mode='reflaction', stride=2, norm='Instance', act="ReLU", drop=False, mode='ConvTranspose')
#         # Denote c7s1-l : d7s1-3(output_channels)
#         self.c7s1_3 = ConvBlock(in_features=64, out_features=output_channels, kernel_size=7, padding=3, padding_mode='reflaction', stride=1, norm=False, act="Tanh", drop=False, mode='Conv')

#     def forward(self, x):
#         x = self.c7s1_64(x)
#         x = self.d128(x)
#         x = self.d256(x)
#         x = self.Rk(x)
#         x = self.u128(x)
#         x = self.u64(x)
#         x = self.c7s1_3(x)    
        
#         return x


# class Discriminator(nn.Module):
#     def __init__(self, input_nc):
#         super(Discriminator, self).__init__()
        
#         # 3 x 256 x 256 -> 64 x 128 x 128
#         self.C64 = ConvBlock(in_features=input_nc, out_features=64, kernel_size=4, padding=1, padding_mode='zeros', stride=2, norm=False, act="LeakyReLU", drop=False, mode='Conv')
#         self.C128 = ConvBlock(in_features=64, out_features=128, kernel_size=4, padding=1, padding_mode='zeros', stride=2, norm="instance", act="LeakyReLU", drop=False, mode='Conv')
#         self.C256 = ConvBlock(in_features=128, out_features=256, kernel_size=4, padding=1, padding_mode='zeros', stride=2, norm="instance", act="LeakyReLU", drop=False, mode='Conv')
#         self.C512 = ConvBlock(in_features=256, out_features=512, kernel_size=4, padding=1, padding_mode='zeros', stride=1, norm="instance", act="LeakyReLU", drop=False, mode='Conv')
#         self.final_layer = ConvBlock(in_features=512, out_features=1, kernel_size=4, padding=1, padding_mode='zeros', stride=1, norm=False, act="LeakyReLU", drop=False, mode='Conv')

#     def forward(self, x):
#         x =  self.C64(x)
#         x =  self.C128(x)
#         x =  self.C256(x)
#         x =  self.C512(x)
#         x =  self.final_layer(x)
#         return x


import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(Generator, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)