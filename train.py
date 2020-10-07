import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pylab
import numpy as np
import torchvision.utils as vutils
from dataloader import data_loader
from arguments import Arguments
from model.gan_network import Generator, Discriminator

class GAN():
    def __init__(self,args):
        self.args = args
        #dataset

        self.data = data_loader
        #Module
        self.G = Generator()
        self.D = Discriminator()
        self.criterion = nn.BCELoss()
        self.optimizerG = optim.Adam(self.G.parameters(), lr=args.lr, betas=(0,0.99))
        self.optimizerD = optim.Adam(self.D.parameters(), lr=args.lr, betas=(0,0.99))
        self.fixed_noise = torch.randn(self.args.batch_size, 100, 1, 1, device=args.device)
    def run(self, save_ckpt=None, load_ckpt=None, result_path=None):
        for epoch in range(args.num_epochs):
            for _iter, data in enumerate(self.data):
                real = data[0].to(self.args.device)
                batch_size = real.size(0)
                label = torch.full((self.args.batch_size,), 1, dtype=real.dtype, device=self.args.device)

                self.D.zero_grad()
                D_real = self.D(real)
                lossD_real = self.criterion(D_real, label)
                lossD_real.backward()
                D_x = D_real.mean().item()

                noise = torch.randn(self.args.batch_size, 100, 1, 1, device=self.args.device)
                
                label.fill_(0)
                
                D_fake = self.D(self.G(noise).detach())
                
                lossD_fake = self.criterion(D_fake, label)
                lossD_fake.backward()
                D_G_z1 = D_fake.mean().item()
                lossD = lossD_real + lossD_fake
                self.optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.G.zero_grad()
                label.fill_(1)

                D_fake = self.D(self.G(noise))
                lossG = self.criterion(D_fake,label)
                lossG.backward()
                D_G_z2 = D_fake.mean().item()
                self.optimizerG.step()

                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                    % (epoch, self.args.num_epochs, _iter, len(self.data),
                        lossD.item(), lossG.item(), D_x, D_G_z1, D_G_z2))
                if _iter % 100 == 0:
                    vutils.save_image(real,
                            '%s/real_samples.png' % result_path,
                            normalize=True)
                    fake = self.G(self.fixed_noise)
                    vutils.save_image(fake.detach(),
                            '%s/fake_samples_epoch_%03d.png' % (result_path, epoch),
                            normalize=True)

    # do checkpointing
        torch.save(self.netG.state_dict(), '%s/netG_epoch_%d.pth' % (save_ckpt, epoch))
        torch.save(self.netD.state_dict(), '%s/netD_epoch_%d.pth' % (save_ckpt, epoch))


if __name__ == '__main__':
    args = Arguments().parser().parse_args()

    args.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')

    # Create a directory if not exists
    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)

    if not os.path.exists(args.result_path):
        os.makedirs(args.ckpt_path)
    model = GAN(args)
    
    model.run(save_ckpt=args.ckpt_path, result_path=args.result_path)



