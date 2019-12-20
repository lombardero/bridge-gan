from __future__ import print_function
import argparse
import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from torch.autograd import Variable
from torchvision.utils import save_image
from models import wgan, wgan_gp, dcgan

def get_model(opt,nc):
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)

    if opt.model == 'WGAN':
        print("Model resolved: WGAN")
        netG = wgan.Generator(ngpu,nz,ngf,nc)
        netD = wgan.Discriminator(ngpu, nc, ndf)
    elif opt.model == 'WGAN-GP':
        print("Model resolved: WGAN_GP")
        netG = wgan_gp.Generator(ngpu,nz, ngf,nc)
        netD = wgan_gp.Discriminator(ngpu, nc, ndf)
    else:
        print("Model resolved: DCGAN")
        netG = dcgan.Generator(ngpu,nz,ngf,nc)
        netD = dcgan.Discriminator(ngpu, nc, ndf)

    return netG, netD

def get_dataloader(opt):
    classes = [ c + '_train' for c in opt.classes.split(',')]
    dataset = dset.LSUN(root=opt.dataroot, classes=classes,
                        transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    opt.nc=3

    print('Generating DataLoader...')
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
    shuffle=True, num_workers=int(opt.workers),pin_memory=opt.cuda)

    return dataloader

def weights_init(m):
    print('Initializing weights')
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
