import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Denote c7s1-k : d7s1
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_channels, 64, kernel_size=7, stride=1),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True) ]

        # Denode dk : d128, d256
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Denote Rk : R256 * 9
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Denote uk : u128, u64
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, kernel_size=4, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Denote c7s1-l : d7s1-3(output_channels)
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_channels, kernel_size=7, stride=1),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

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


