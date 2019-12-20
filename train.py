from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.autograd as autograd

from torch.autograd import Variable
from torchvision.utils import save_image

def train_wgan(opt,netG,netD, dataloader, device):
    fixed_noise = torch.randn(opt.batchSize, int(opt.nz), 1, 1, device=device)
    
    optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lr)
    optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lr)

    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor

    batches_done = 0
    for epoch in range(opt.niter):
        for i, (imgs, _) in enumerate(dataloader,0):
            # Configure input
            real_imgs = Variable(imgs.type(Tensor))
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizerD.zero_grad()
            # Sample noise as generator input
            z = torch.randn(opt.batchSize, int(opt.nz), 1, 1, device=device)
            # Generate a batch of images
            fake_imgs = netG(z).detach()
            # Adversarial loss
            loss_D = -torch.mean(netD(real_imgs)) + torch.mean(netD(fake_imgs))
            loss_D.backward()
            optimizerD.step()
            # Clip weights of discriminator
            for p in netD.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)
            # Train the generator every n_critic iterations
            if i % opt.n_critic == 0:
                #  Train Generator
                optimizerG.zero_grad()
                # Generate a batch of images
                gen_imgs = netG(z)
                # Adversarial loss
                loss_G = -torch.mean(netD(gen_imgs))
                loss_G.backward()
                optimizerG.step()
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.niter, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
                )

            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], "%s/%d.png" % (opt.outimg,batches_done), nrow=5, normalize=True)
            batches_done += 1

        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))

def compute_gradient_penalty(D, real_samples, fake_samples, device, opt):
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor

    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1)
    alpha = alpha.expand(real_samples.size(0), int(real_samples.nelement()/real_samples.size(0))).contiguous()
    alpha = alpha.view(real_samples.size(0), 3, opt.imageSize, opt.imageSize)
    alpha = alpha.to(device)
    # Resize the fake_samples
    fake_samples = fake_samples.view(real_samples.size(0), 3, opt.imageSize, opt.imageSize)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(d_interpolates.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs = d_interpolates,
        inputs = interpolates,
        grad_outputs=torch.ones(d_interpolates.size()).to(device),
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_wgan_gp(opt, netG, netD, dataloader, device):
    fixed_noise = torch.randn(opt.batchSize, int(opt.nz), 1, 1, device=device)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor
    batches_done = opt.startiter

    for epoch in range(opt.startepoch, opt.niter):
        for i, (imgs, _) in enumerate(dataloader,0):
            # Configure input
            real_imgs = Variable(imgs.type(Tensor))
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizerD.zero_grad()
            # Sample noise as generator input
            z = torch.randn(real_imgs.shape[0], int(opt.nz), 1, 1, device=device)
            # Generate a batch of images
            fake_imgs = netG(z).detach()
            # Adversarial losss with Gradient Penalty
            fake_score = netD(fake_imgs)
            real_score = netD(real_imgs)
            gradient_penalty = compute_gradient_penalty(netD,real_imgs.data,fake_imgs.data,device,opt)
            
            loss_D = -torch.mean(netD(real_imgs)) + torch.mean(netD(fake_imgs)) + opt.lambda_gp * gradient_penalty
            loss_D.backward()
            optimizerD.step()

            # Train the generator every n_critic iterations
            if i % opt.n_critic == 0:
                #  Train Generator
                optimizerG.zero_grad()
                # Generate a batch of images
                gen_imgs = netG(z)
                # Adversarial loss
                loss_G = -torch.mean(netD(gen_imgs))
                loss_G.backward()
                optimizerG.step()
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [W distance: %f]"
                    % (epoch, opt.niter, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item(), torch.mean(fake_score)-torch.mean(real_score))
                )

            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], "%s/%d.png" % (opt.outimg,batches_done), nrow=5, normalize=True)
            batches_done += 1

        torch.save(netG.state_dict(), '%s/netG_epoch_%d_%d.pth' % (opt.outf, epoch, batches_done))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d_%d.pth' % (opt.outf, epoch, batches_done))

def train_dcgan(opt, netG, netD, dataloader, device):
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(opt.batchSize, int(opt.nz), 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    for epoch in range(opt.startiter, opt.niter):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device)

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, int(opt.nz), 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()


            if i % 100 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                    % (epoch, opt.niter, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                vutils.save_image(real_cpu,
                        '%s/real_samples.png' % opt.outf,
                        normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(),
                        '%s/fake_samples_epoch_%03d.png' % (opt.outimg, epoch),
                        normalize=True)

        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))