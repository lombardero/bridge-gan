# Bridge-GAN

This is a pytorch implementation of 3 different GAN models on LSUN-Bridge dataset.
1. DCGAN
2. WGAN
3. WGAN with Gradient Penalty

![Alt text](result.png?raw=true "Title")

## Dataset
This implementation is for LSUN Bridge dataset.
Clone the repo: https://github.com/fyu/lsun.git

Download the whole latest data set
python3 download.py

Download the whole latest data set to <data_dir>
python3 download.py -o <data_dir>

Download data for bedroom
python3 download.py -c bedroom

Download testing set
python3 download.py -c test

Detailed instructions: https://github.com/fyu/lsun

## Training

python main.py --dataroot PATH TO DATASET --outf PATH FOR SAVING TRAINED MODEL --outimg PATH FOR SAVING IMAGES --niter EPOCHS --model PICK ONE: 'WGAN','WGAN_GP','DCGAN'

Use this to resume training:

python main.py --dataroot PATH TO DATASET --outf PATH FOR SAVING TRAINED MODEL --outimg PATH FOR SAVING IMAGES --niter EPOCHS --model PICK ONE: 'WGAN','WGAN_GP','DCGAN' --netG Path to trained generator --netD Path to trained discriminator

## Trained Models
This repository contains 5 trained models with following config:
1. WGAN - 25 epochs , 40 epochs
2. WGAN_GP - 25 epochs
3. DCGAN - 25 epochs , 40 epochs
