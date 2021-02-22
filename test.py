import argparse
import itertools
import os, sys

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import itertools

from utils import *
from model import Generator, Discriminator
from dataloader import data_loader


def test(args):
    # network init
    netG_A2B = Generator(args.input_nc, args.output_nc, args.n_Rk).to(device=args.device)
    netG_B2A = Generator(args.output_nc, args.input_nc, args.n_Rk).to(device=args.device)
    args.result_path = args.result_path+"_test"
    try:
        ckpt = load_checkpoint(args.ckpt_path, args.device)
        netG_B2A.load_state_dict(ckpt['netG_B2A'])
        netG_A2B.load_state_dict(ckpt['netG_A2B'])
    except:
        print('Failed to load checkpoint')

    dataloader = data_loader(args, mode = 'test')
    netG_B2A.eval()
    netG_A2B.eval()
    for i in range(len(dataloader)):
       sample_images(args, i, self.netG_A2B, self.netG_B2A, dataloader)