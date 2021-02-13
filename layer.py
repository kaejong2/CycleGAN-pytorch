import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, padding_mode='zeros', stride=1, norm=False, act=False, drop=False, mode='conv'):
        super(ConvBlock, self).__init__()

        block = []
        if mode=='Conv':
            if padding_mode == 'reflaction':
                block += [nn.ReflectionPad2d(padding)]
            elif padding_mode == 'zeros':
                block += [nn.ZeroPad2d(padding)]
            block += [nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride)]
            
        elif mode=='ConvTranspose':
            block += [nn.ConvTranspose2d(in_features, out_features, kernel_size=kernel_size, padding=padding, output_padding=0, stride=stride)]
            # if padding_mode == 'reflaction':
            #     block += [nn.ReflectionPad2d(padding)]
            # elif padding_mode == 'zeros':
            #     block += [nn.ZeroPad2d(padding)]

        if not norm is False:
            if norm == "Batch":
                block += [nn.BatchNorm2d(out_features)]
            elif norm == "Instance":
                block += [nn.InstanceNorm2d(out_features)]
        if not act is False:
            if act == "ReLU":
                block += [nn.ReLU(inplace=True)]
            elif act == "LeakyReLU":
                block += [nn.LeakyReLU(0.2, inplace=True)]
            elif act == "Tanh":
                block += [nn.Tanh()]
            elif act == "Sigmoid":
                block += [nn.Sigmoid()]
        if not drop is False:
            block += [nn.Dropout2d(0.5)]
        
        self.layer = nn.Sequential(*block)

    def forward(self, x):
        return self.layer(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, padding_mode='zeros', stride=1, norm=False, act=False, drop=False, mode='conv' ):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = ConvBlock(in_features=in_features, out_features=out_features, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, stride=stride, norm=norm, act=act, drop=False, mode='Conv')
        self.conv2 = ConvBlock(in_features=in_features, out_features=out_features, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, stride=stride, norm=norm, act=False, drop=False, mode='Conv')

    def forward(self, x):
        res_x = self.conv1(x)
        res_x = self.conv2(x)
        return x + res_x