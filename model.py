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
        model =  [  nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]
        # 64 x 128 x 128 -> 128 x 64 x 64
        model += [  nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]
        # 128 x 64 x 64 -> 256 x 32 x 32
        model += [  nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]
        # 256 x 32 x 32 -> 512 x 16 x 16
        model += [  nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]
        # 512 x 16 x 16 -> 1 x 16 x 16
        model += [nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return x
        # return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


