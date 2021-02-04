import os, sys
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import itertools

from utils import *
from model import Generator, Discriminator
from dataloader import data_loader





def test(self, args):
    # network init
    netG_A2B = Generator(self.args.input_nc, self.args.output_nc).to(device= self.args.device)
    netG_B2A = Generator(self.args.output_nc, self.args.input_nc).to(device= self.args.device)
    
    try:
        ckpt = load_checkpoint(args.ckpt_path)
        netG_B2A.load_state_dict(ckpt['netG_B2A'])
        netG_A2B.load_state_dict(ckpt['netG_A2B'])
    except:
        print('Failed to load checkpoint')

    dataloader = data_loader(self.args, mode = 'test')
    data = iter(dataloader).next()[0]
    real_A = data['img_A'].to(device=args.device)
    real_B = data['img_B'].to(device=args.device)
    
    netG_B2A.eval()
    netG_A2B.eval()
    with torch.no_grad():
        fake_A = netG_A2B(real_A)
        fake_B = netG_B2A(real_B)
        recon_A = netG_A2B(fake_B)
        recon_B = netG_B2A(fake_A)

    result = (torch.cat([real_A, fake_B, recon_A, real_B, fake_A, recon_B], dim=0).data + 1)/ 2.0

    torchvision.utils.save_image(result, args.result_path+'sample.jpg', nrow=3)
