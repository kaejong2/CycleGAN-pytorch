import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import ConvBlock, ResidualBlock


class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Denote c7s1-k : d7s1
        self.c7s1_64 = ConvBlock(in_features=input_channels, out_features=64, kernel_size=7, padding=3, padding_mode='reflaction', stride=1, norm='Instance', act="ReLU", drop=False, mode='Conv')

        # Denode dk : d128, d256
        self.d128 = ConvBlock(in_features=64, out_features=128, kernel_size=4, padding=1, padding_mode='reflaction', stride=2, norm='Instance', act="ReLU", drop=False, mode='Conv')
        self.d256 = ConvBlock(in_features=128, out_features=256, kernel_size=4, padding=1, padding_mode='reflaction', stride=2, norm='Instance', act="ReLU", drop=False, mode='Conv')
        
        # Denote Rk : R256 * 9
        res = []
        for _ in range(n_residual_blocks):
            res += [ResidualBlock(256, 256, kernel_size=3, padding=1, padding_mode='reflaction', stride=1, norm='instance', act='ReLU', drop=False, mode='conv')]
        self.Rk = nn.Sequential(*res)

        # Denote uk : u128, u64
        self.u128 = ConvBlock(in_features=256, out_features=128, kernel_size=4, padding=1, padding_mode='zeros', stride=2, norm='Instance', act="ReLU", drop=False, mode='ConvTranspose')
        self.u64  = ConvBlock(in_features=128, out_features=64, kernel_size=4, padding=1, padding_mode='reflaction', stride=2, norm='Instance', act="ReLU", drop=False, mode='ConvTranspose')
        model = [  nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                        nn.InstanceNorm2d(128),
                        nn.ReLU(inplace=True) ]
        self.model = nn.Sequential(*model)
        # Denote c7s1-l : d7s1-3(output_channels)
        self.c7s1_3 = ConvBlock(in_features=64, out_features=output_channels, kernel_size=7, padding=3, padding_mode='reflaction', stride=1, norm=False, act="Tanh", drop=False, mode='Conv')

    def forward(self, x):
        x = self.c7s1_64(x)
        x = self.d128(x)
        x = self.d256(x)
        x = self.Rk(x)
        x = self.u128(x)
        x = self.u64(x)
        x = self.c7s1_3(x)    
        
        return x

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        
        # 3 x 256 x 256 -> 64 x 128 x 128
        self.C64 = ConvBlock(in_features=input_nc, out_features=64, kernel_size=4, padding=1, padding_mode='zeros', stride=2, norm=False, act="LeakyReLU", drop=False, mode='Conv')
        self.C128 = ConvBlock(in_features=64, out_features=128, kernel_size=4, padding=1, padding_mode='zeros', stride=2, norm=False, act="LeakyReLU", drop=False, mode='Conv')
        self.C256 = ConvBlock(in_features=128, out_features=256, kernel_size=4, padding=1, padding_mode='zeros', stride=2, norm=False, act="LeakyReLU", drop=False, mode='Conv')
        self.C512 = ConvBlock(in_features=256, out_features=512, kernel_size=4, padding=1, padding_mode='zeros', stride=1, norm=False, act="LeakyReLU", drop=False, mode='Conv')
        self.final_layer = ConvBlock(in_features=512, out_features=1, kernel_size=4, padding=1, padding_mode='zeros', stride=1, norm=False, act="LeakyReLU", drop=False, mode='Conv')
        


    def forward(self, x):
        x =  self.C64(x)
        x =  self.C128(x)
        x =  self.C256(x)
        x =  self.C512(x)
        x =  self.final_layer(x)
        return x



