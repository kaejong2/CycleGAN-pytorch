
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import *



import argparse
import itertools
import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import torch.optim as optim
from Model.cycleGAN import Generator, Discriminator

from utils import ReplayBuffer

from utils import Logger
from utils import weights_init_normal
from utils import save
from utils import set_requires_grad

import os

from dataloader import data_loader

from arguments import Arguments




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
