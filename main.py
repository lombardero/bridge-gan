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
import utils
import train
from torch.autograd import Variable
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--startiter', type=int, default=0, help='iteration number model is starting at (if model re-loaded)')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='checkpoint', help='folder to output images and model checkpoints')
parser.add_argument('--outimg', default='images', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--classes', default='bridge', help='comma separated list of classes for the lsun data set')
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--fid_real", type=int, default=0, help="interval betwen image samples")
parser.add_argument("--fid_fake", type=int, default=0, help="interval betwen image samples")

opt = parser.parse_args()
print(opt)

os.makedirs(opt.outimg, exist_ok=True)
os.makedirs(opt.outf, exist_ok=True)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available():
    opt.cuda = True
    print('**** GPU enabled ****')
print('loading dataset...')

nc=3
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)

dataloader = utils.get_dataloader(opt)
device = torch.device("cuda:0" if opt.cuda else "cpu")

netG, netD = utils.get_model(opt,nc)
netG.apply(utils.weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD.apply(utils.weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

if opt.model == 'WGAN':
        train.train_wgan(opt,netG,netD,dataloader,device)
elif opt.model == 'WGAN-GP':
    train.train_wgan_gp(opt,netG,netD,dataloader,device)
else:
    train.train_dcgan(opt,netG,netD,dataloader,device)

