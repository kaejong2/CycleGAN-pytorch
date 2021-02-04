
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



class cycleGAN():
    def __init__(self, args):
        self.args = args
        
        # network init
        self.netG_A2B = Generator(self.args.input_nc, self.args.output_nc).to(device= self.args.device)
        self.netG_B2A = Generator(self.args.output_nc, self.args.input_nc).to(device= self.args.device)
        self.netD_A = Discriminator(self.args.input_nc).to(device= self.args.device)
        self.netD_B = Discriminator(self.args.output_nc).to(device= self.args.device)
        
        self.netG_A2B.apply(weights_init_normal)
        self.netG_B2A.apply(weights_init_normal)
        self.netD_A.apply(weights_init_normal)
        self.netD_B.apply(weights_init_normal)
        
        # Loss
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()
        
        # optimizer
        self.optimizerG = optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()), lr=self.args.lr, betas=(self.args.b1,self.args.b2))
        self.optimizerD = optim.Adam(itertools.chain(self.netD_A.parameters(),self.netD_B.parameters()), lr=self.args.lr, betas=(self.args.b1,self.args.b2))


        # Learning rate scheduler
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizerG, lr_lambda=LambdaLR(self.args.num_epochs, self.args.epoch, self.args.decay_epoch).step)
        self.lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.optimizerD, lr_lambda=LambdaLR(self.args.num_epochs, self.args.epoch, self.args.decay_epoch).step)
        
        #dataset
        self.dataloader = data_loader(self.args)

        # plot
        # self.logger = Logger(self.args.num_epochs, len(self.dataloader))
    

    def run(self, save_ckpt=None, load_ckpt=None, result_path=None):
        for epoch in range(self.args.num_epochs):
            self.netG_A2B.train()
            self.netG_B2A.train()
            self.netD_A.train()
            self.netD_B.train()
            
            loss_G_A2B_train = []
            loss_G_B2A_train = []
            loss_D_A_train = []
            loss_D_B_train = []

            loss_cycle_A_train = []
            loss_cycle_B_train = []
            loss_identity_A_train = []
            loss_identity_B_train = []
            
            for _iter, data in enumerate(self.dataloader):
                real_A = data['A'].to(device= self.args.device)
                real_B = data['B'].to(device= self.args.device)

                output_B = self.netG_A2B(real_A)
                recon_A = self.netG_B2A(output_B)

                output_A = self.netG_B2A(real_B)
                recon_B = self.netG_A2B(output_A)

                #Discriminator Loss
                set_requires_grad([self.netD_A, self.netD_B], True)

                self.optimizerD.zero_grad()
                
                pred_real_A = self.netD_A(real_A)
                pred_fake_A = self.netD_A(output_A.detach())

                loss_D_A_real = self.criterion_GAN(pred_real_A, torch.ones_like(pred_real_A))
                loss_D_A_fake = self.criterion_GAN(pred_fake_A, torch.zeros_like(pred_fake_A))
                loss_D_A = 0.5 * ( loss_D_A_real + loss_D_A_fake)

                pred_real_B = self.netD_B(real_B)
                pred_fake_B = self.netD_B(output_B.detach())

                loss_D_B_real = self.criterion_GAN(pred_real_B, torch.ones_like(pred_real_B))
                loss_D_B_fake = self.criterion_GAN(pred_fake_B, torch.zeros_like(pred_fake_B))
                loss_D_B = 0.5 * ( loss_D_B_real + loss_D_B_fake)

                loss_D = loss_D_A + loss_D_A
                
                loss_D.backward()
                
                self.optimizerG.zero_grad()

                #Generator loss
                set_requires_grad([self.netD_A, self.netD_B], False)
                
                pred_fake_A = self.netD_A(output_A)
                pred_fake_B = self.netD_B(output_B)

                # Adversarial Loss
                loss_G_A2B = self.criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))
                loss_G_B2A = self.criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B))
                # cycle consistancy loss
                loss_cycle_A = self.criterion_cycle(recon_A, real_A) * 10
                loss_cycle_B = self.criterion_cycle(recon_B, real_B) * 10
                # itentity loss
                identity_A = self.netG_B2A(real_A)
                identity_B = self.netG_A2B(real_B)

                loss_identity_A = self.criterion_identity(real_A, identity_A) * 5
                loss_identity_B = self.criterion_identity(real_B, identity_B) * 5

                loss_G = loss_G_A2B + loss_G_B2A + loss_cycle_A + loss_cycle_B + loss_identity_A + loss_identity_B

                loss_G.backward()

                self.optimizerG.step()

                loss_G_A2B_train += [loss_G_A2B.item()]
                loss_G_B2A_train += [loss_G_B2A.item()]
                
                loss_D_A_train += [loss_D_A.item()]
                loss_D_B_train += [loss_D_B.item()]
                
                loss_cycle_A_train += [loss_cycle_A.item()]
                loss_cycle_B_train += [loss_cycle_B.item()]
                
                loss_identity_A_train += [loss_identity_A.item()]
                loss_identity_B_train += [loss_identity_B.item()]

               
                print("Train : Epoch %04d/ %04d | Batch %04d / %04d | "
                       "Generator A2B %.4f B2A %.4f | "
                       "Discriminator A %.4f B %.4f | "
                       "Cycle A %.4f B %.4f | "
                       "Identity A %.4f B %.4f | " % 
                       (epoch, self.args.num_epochs, _iter, len(self.dataloader),
                       np.mean(loss_G_A2B_train), np.mean(loss_G_B2A_train),
                       np.mean(loss_D_A_train), np.mean(loss_D_B_train),
                       np.mean(loss_cycle_A_train), np.mean(loss_cycle_B_train),
                       np.mean(loss_identity_A_train), np.mean(loss_identity_B_train)))
            

            save(save_ckpt, self.netG_A2B, self.netG_B2A, self.netD_A, self.netD_B, self.optimizerG, self.optimizerD, epoch)
        


        


