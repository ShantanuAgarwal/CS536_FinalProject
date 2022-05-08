import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from datetime import date
from models import Generator, Discriminator
import cv2
import os
import os.path
import pickle
import torchvision
from torch.autograd import Variable
import random
from torchvision.utils import save_image



# Utils
import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
from visdom import Visdom
import numpy as np

def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)
        

import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB'))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('RGB'))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
        
import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

# Hyperparameters
start_epoch = 0
n_epochs = 50
batchSize = 1
dataroot = 'datasets/real2pre_recorded'
lr = 0.0002
decay_epoch = 30
size = 256
input_nc = 3
output_nc = 3
cuda = True
n_gpu = 1
n_cpu = 8
resume = False
resume_path = "output/checkpoint_latest.pth"
if torch.cuda.is_available() and not cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
netG_A2B = Generator(input_nc, output_nc).cuda().float()
netG_B2A = Generator(output_nc, input_nc).cuda().float()
netD_A = Discriminator(input_nc).cuda().float()
netD_B = Discriminator(output_nc).cuda().float()
    
if not resume:
        netG_A2B.apply(weights_init_normal)
        netG_B2A.apply(weights_init_normal)
        netD_A.apply(weights_init_normal)
        netD_B.apply(weights_init_normal)
else:
    checkpoint = torch.load(resume_path)
    netG_A2B.load_state_dict(checkpoint['netG_A2B'])
    netG_B2A.load_state_dict(checkpoint['netG_B2A'])
    netD_A.load_state_dict(checkpoint["netD_A"])
    netD_B.load_state_dict(checkpoint["netD_B"])
    start_epoch = checkpoint['epoch']



criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, start_epoch, decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(n_epochs, start_epoch, decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(n_epochs, start_epoch, decay_epoch).step)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor


input_A = Tensor(batchSize, input_nc, size, size)
input_B = Tensor(batchSize, output_nc, size, size)
target_real = Variable(torch.ones(batchSize).to(device))
target_fake = Variable(torch.zeros(batchSize).to(device))

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()



transforms_ = [ transforms.Resize(int(size*1.12), transforms.InterpolationMode.BICUBIC), 
                transforms.RandomCrop(size), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]



dataloader = torch.utils.data.DataLoader(ImageDataset(dataroot, transforms_ = transforms_, unaligned=True), 
                        batch_size=batchSize, shuffle=True, num_workers=n_cpu, drop_last=True)

dataloader_test = torch.utils.data.DataLoader(ImageDataset(dataroot, transforms_=transforms_, mode='test'), 
                        batch_size=batchSize, shuffle=False, num_workers=n_cpu, drop_last=True)




G_losses = []
G_losses_identity = []
G_losses_GAN = []
G_losses_cycle = []
D_losses = []
print("Training started")
from tqdm import tqdm
for i in tqdm(range(start_epoch, n_epochs)):
    for j, batch in tqdm(enumerate(dataloader)):
        netG_A2B.train()
        netG_B2A.train()
        netD_A.train()
        netD_B.train()
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))
        real_A = real_A.to(device)
        real_B = real_B.to(device)
        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(torch.squeeze(pred_fake, 1), target_real)


        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(torch.squeeze(pred_fake, 1), target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()

        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(torch.squeeze(pred_real, 1), target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(torch.squeeze(pred_fake, 1), target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(torch.squeeze(pred_real, 1), target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(torch.squeeze(pred_fake, 1), target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################
        save_object =  {
            'loss_G': loss_G.item(),
            'loss_G_identity' : loss_identity_A.item() + loss_identity_B.item(),
            'loss_G_GAN': loss_GAN_A2B.item() + loss_GAN_B2A.item(),
            'loss_G_cycle': loss_cycle_ABA.item() + loss_cycle_BAB.item(),
            'loss_D': loss_D_A.item() + loss_D_A.item(),
            'real_A': real_A,
            'real_B': real_B,
            'fake_A': fake_A,
            'fake_B': fake_B,
            'recovered_A': recovered_A,
            'recovered_B': recovered_B
        }
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    A_output_imgs = []
    A_output_imgs.extend(save_object['real_A'])
    A_output_imgs.extend(save_object['fake_B'])
    A_output_imgs.extend(save_object['recovered_A'])
    B_output_imgs = []
    B_output_imgs.extend(save_object['real_B'])
    B_output_imgs.extend(save_object['fake_A'])
    B_output_imgs.extend(save_object['recovered_B'])
    save_image(A_output_imgs, f'output_images/images_A_{i}.png')
    save_image(B_output_imgs, f'output_images/images_B_{i}.png')
    if i % 5 == 0:
        weights = {}
        weights['netG_A2B'] = netG_A2B.state_dict()
        weights['netG_B2A'] = netG_B2A.state_dict()
        weights['netD_A'] = netD_A.state_dict()
        weights['netD_B'] = netD_B.state_dict()
        weights['epoch'] = i
        torch.save(weights, f'output/checkpoint_{i}.pth')