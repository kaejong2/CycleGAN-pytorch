
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


class cycleGAN():
    def __init__(self, args):
        self.args = args
        
        # Define the Network
        # self.netG_A2B = Generator(input_channels=self.args.input_nc, output_channels=self.args.output_nc, n_residual_blocks=self.args.n_Rk).to(device=self.args.device)
        self.netG_A2B = Generator(input_shape=(3,256,256), num_residual_blocks=self.args.n_Rk).to(device=self.args.device)
        # self.netG_B2A = Generator(input_channels=self.args.output_nc, output_channels=self.args.input_nc, n_residual_blocks=self.args.n_Rk).to(device=self.args.device)
        self.netG_B2A = Generator(input_shape=(3,256,256), num_residual_blocks=self.args.n_Rk).to(device=self.args.device)
        # self.netD_A = Discriminator(input_nc=self.args.input_nc).to(device=self.args.device)
        # self.netD_B = Discriminator(input_nc=self.args.output_nc).to(device=self.args.device)        
        self.netD_A = Discriminator(input_shape=(3,256,256)).to(device=self.args.device)    
        self.netD_B = Discriminator(input_shape=(3,256,256)).to(device=self.args.device)        
        init_weight(self.netD_B, init_type=args.init_weight)
        init_weight(self.netD_A, init_type=args.init_weight)
        init_weight(self.netG_A2B, init_type=args.init_weight)        
        init_weight(self.netG_B2A, init_type=args.init_weight)

        # Define Loss function
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()
        
        # Optimizers
        self.optimizerG = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()), lr=self.args.lr, betas=(self.args.b1, self.args.b2))
        self.optimizerD_A = torch.optim.Adam(self.netD_A.parameters(), lr=self.args.lr, betas=(self.args.b1, self.args.b2))
        self.optimizerD_B = torch.optim.Adam(self.netD_B.parameters(), lr=self.args.lr, betas=(self.args.b1, self.args.b2))

        # Learning rate scheduler
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizerG, lr_lambda=LambdaLR(self.args.num_epochs, self.args.epoch, self.args.decay_epoch).step)
        self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(self.optimizerD_A, lr_lambda=LambdaLR(self.args.num_epochs, self.args.epoch, self.args.decay_epoch).step)
        self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(self.optimizerD_B, lr_lambda=LambdaLR(self.args.num_epochs, self.args.epoch, self.args.decay_epoch).step)
        
        #dataset
        self.dataloader = data_loader(self.args)
        self.val_dataloader = data_loader(self.args, mode='test')

    def run(self):
        self.netG_A2B.train()
        self.netG_B2A.train()
        self.netD_A.train()
        self.netD_B.train()
        for epoch in range(self.args.num_epochs):   
            loss_G_A2B_train = []
            loss_G_B2A_train = []
            loss_D_A_train = []
            loss_D_B_train = []
            loss_cycle_A_train = []
            loss_cycle_B_train = []
            loss_identity_A_train = []
            loss_identity_B_train = []
            
            for _iter, imgs in enumerate(self.dataloader):
                real_A = imgs['img_A'].to(device=self.args.device)
                real_B = imgs['img_B'].to(device=self.args.device)

                fake_B = self.netG_A2B(real_A)
                fake_A = self.netG_B2A(real_B)

                recon_A = self.netG_B2A(fake_B)
                recon_B = self.netG_A2B(fake_A)

                identity_A = self.netG_B2A(real_A)
                identity_B = self.netG_A2B(real_B)

                #################################################
                #              Train Discriminator
                #################################################

                self.optimizerD_A.zero_grad()
                
                real_A_dis = self.netD_A(real_A)
                fake_A_dis = self.netD_A(fake_A.detach())

                loss_D_A_real = self.criterion_GAN(real_A_dis, torch.ones_like(real_A_dis))
                loss_D_A_fake = self.criterion_GAN(fake_A_dis, torch.zeros_like(fake_A_dis))
                loss_D_A = (loss_D_A_real + loss_D_A_fake)/2
                
                loss_D_A.backward()
                self.optimizerD_A.step()

                self.optimizerD_B.zero_grad()
                real_B_dis = self.netD_B(real_B)
                fake_B_dis = self.netD_B(fake_B.detach())

                loss_D_B_real = self.criterion_GAN(real_B_dis, torch.ones_like(real_B_dis))
                loss_D_B_fake = self.criterion_GAN(fake_B_dis, torch.zeros_like(fake_B_dis))
                loss_D_B = (loss_D_B_real + loss_D_B_fake)/2
                
                loss_D_B.backward()
                self.optimizerD_B.step()


                #################################################
                #              Train Generator
                #################################################

                self.optimizerG.zero_grad()
                
                fake_A_dis = self.netD_A(fake_A)
                fake_B_dis = self.netD_B(fake_B)
                # Adversarial Loss
                loss_G_A2B = self.criterion_GAN(fake_A_dis, torch.ones_like(fake_A_dis))
                loss_G_B2A = self.criterion_GAN(fake_B_dis, torch.ones_like(fake_B_dis))
                loss_G = (loss_G_A2B + loss_G_B2A)/2
                # cycle consistancy loss
                loss_cycle_A = self.criterion_cycle(recon_A, real_A)
                loss_cycle_B = self.criterion_cycle(recon_B, real_B)
                loss_cycle = (loss_cycle_A + loss_cycle_B)/2
                # itentity loss
                loss_identity_A = self.criterion_identity(real_A, identity_A)
                loss_identity_B = self.criterion_identity(real_B, identity_B)
                loss_identity = (loss_identity_A + loss_identity_B)/2

                loss_G = loss_G + loss_cycle*10 + loss_identity*5
                
                loss_G.backward()
                self.optimizerG.step()
                #################################################
                #                   Log Progress
                #################################################
                loss_G_A2B_train += [loss_G_A2B.item()]
                loss_G_B2A_train += [loss_G_B2A.item()]
                
                loss_D_A_train += [loss_D_A.item()]
                loss_D_B_train += [loss_D_B.item()]
                
                loss_cycle_A_train += [loss_cycle_A.item()]
                loss_cycle_B_train += [loss_cycle_B.item()]
                
                loss_identity_A_train += [loss_identity_A.item()]
                loss_identity_B_train += [loss_identity_B.item()]

                batches_done = epoch * len(self.dataloader) + _iter
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
                if batches_done % self.args.sample_save ==0:
                    print("sample save")
                    sample_images(self.args, batches_done, self.netG_A2B, self.netG_B2A, self.val_dataloader)
            
            save(os.path.join(self.args.root_path, self.args.ckpt_path), self.netG_A2B, self.netG_B2A, self.netD_A, self.netD_B, self.optimizerG, self.optimizerD_A, self.optimizerD_B, epoch)
        
            self.lr_scheduler_G.step()
            self.lr_scheduler_D_A.step()
            self.lr_scheduler_D_B.step()
        
        


