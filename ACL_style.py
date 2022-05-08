# Ignite
import torch
import torchvision
import ignite
import pickle

print(*map(lambda m: ": ".join((m.__name__, m.__version__)), (torch, torchvision, ignite)), sep="\n")

import os
import logging
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.utils as vutils

from ignite.engine import Engine, Events
import ignite.distributed as idist

ignite.utils.manual_seed(999)

ignite.utils.setup_logger(name="ignite.distributed.auto.auto_dataloader", level=logging.WARNING)
ignite.utils.setup_logger(name="ignite.distributed.launcher.Parallel", level=logging.WARNING)

import torchvision.datasets as dset
from torchvision.utils import save_image

# Model
import torch.nn as nn
import torch.nn.functional as F
import math

from D_hat import MsImageDis
from model_attn_v2 import Generator as Gen, Discriminator as Dis



image_save_iter= 10000         # How often do you want to save output images during training
image_display_iter= 1000       # How often do you want to display output images during training
display_size= 16              # How many images do you want to display each time
snapshot_save_iter= 10000      # How often do you want to save trained models
log_iter= 1                   # How often do you want to log the training stats

# optimization options
max_iter= 350000              # maximum number of training iterations
batch_size= 1                 # batch size
weight_decay= 0.0001          # weight decay
beta1= 0.5                    # Adam parameter
beta2= 0.999                  # Adam parameter
init= 'kaiming'                 # initialization [gaussian/kaiming/xavier/orthogonal]
lr_1= 0.0001                    # initial learning rate
lr_policy= 'step'               # learning rate scheduler
step_size= 100000             # how often to decay learning rate
gamma= 0.5                    # how much to decay learning rate
gan_w= 1                      # weight of adversarial loss
gan_cw= 0.2                   # weight of council loss
recon_x_w= 1                  # weight of image reconstruction loss
recon_s_w= 1                  # weight of style reconstruction loss
recon_c_w= 1                  # weight of content reconstruction loss
recon_x_cyc_w= 1              # weight of explicit style augmented cycle consistency loss
vgg_w= 0                      # weight of domain-invariant perceptual loss
alpha= 1                      # weight of z in council loss
G_update= 2                   # weight of z in council loss
D_update= 1                   # weight of z in council loss

gen={"dim": 64,                     # number of filters in the bottommost layer
  "mlp_dim": 256,                # number of filters in MLP
  "style_dim": 8,                # length of style code
  "output_dim": 4,               # length of style code
  "activ": 'relu',                 # activation function [relu/lrelu/prelu/selu/tanh]
  "n_downsample": 2,             # number of downsampling layers in content encoder
  "n_res": 4,                    # number of residual blocks in content encoder/decoder
  "pad_type": 'reflect' }          # padding type [zero/reflect]
dis={"dim": 64,                     # number of filters in the bottommost layer
  "norm": 'none',                  # normalization layer [none/bn/in/ln]
  "activ": 'lrelu',                # activation function [relu/lrelu/prelu/selu/tanh]
  "n_layer": 4,                  # number of layers in D
  "gan_type": 'lsgan',             # GAN loss [lsgan/nsgan]
  "num_scales": 3,               # number of scales
  "pad_type": 'reflect'}          # padding type [zero/reflect]

# data options
input_dim_a= 3                              # number of image channels [1/3]
input_dim_b= 6                              # number of image channels [1/3]
num_workers= 8                              # number of data loading threads
new_size= 256                               # first resize the shortest image side to this size
crop_image_height= 256                      # random crop image of this height
crop_image_width= 256                       # random crop image of this width
dataroot = 'datasets/real2pre_recorded'
data_kind: "real2pre_recorded"
gan_type = 'lsgan'
# Residual block

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
    if classname.find('Conv') != -1 and classname !='Conv2dBlock':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
# Dataset
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

# Train
import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

# Hyperparameters
epoch = 10
n_epochs = 200
batchSize = 1
lr = 0.0002
decay_epoch = 35
size = 256
input_nc = 3
output_nc = 3
cuda = True
n_gpu = 4
n_cpu = 8
resume = False

gan_curriculum = 10
starting_rate = 0.01
default_rate = 0.5
lambda_reg_attn = 1e-6

lambda_real = 1
lambda_fake = 0
lambda_step = lambda_real/(2*n_epochs)

ROTATION_DEGREE = 10

random_rotator = transforms.RandomRotation(degrees=(-ROTATION_DEGREE, ROTATION_DEGREE))

if torch.cuda.is_available() and not cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

netG_A2B = idist.auto_model(Gen())
netG_B2A = idist.auto_model(Gen())
netD_A = idist.auto_model(Dis())
netD_B = idist.auto_model(Dis())
dis_2 = idist.auto_model(MsImageDis(input_dim_b, dis))
    
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
criterion_acl = torch.nn.MSELoss()
criterion_mse = torch.nn.MSELoss()
optimizer_G = idist.auto_optim(torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=lr, betas=(0.5, 0.999)))
optimizer_D_A = idist.auto_optim(torch.optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.999)))
optimizer_D_B = idist.auto_optim(torch.optim.Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.999)))
optimizer_D_2 = idist.auto_optim(torch.optim.Adam(dis_2.parameters(), lr=lr_1, betas=(0.5, 0.999), weight_decay=weight_decay))

lr_scheduler_G = idist.auto_optim(torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step))
lr_scheduler_D_A = idist.auto_optim(torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step))
lr_scheduler_D_B = idist.auto_optim(torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

input_A = Tensor(batchSize, input_nc, size, size)
input_B = Tensor(batchSize, output_nc, size, size)
target_real = Variable(Tensor(batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(batchSize).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

transforms_ = [ transforms.Resize(int(size*1.12), transforms.InterpolationMode.BICUBIC), 
                transforms.RandomCrop(size), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

dataloader = idist.auto_dataloader(ImageDataset(dataroot, transforms_ = transforms_, unaligned=True), 
                        batch_size=batchSize, shuffle=True, num_workers=n_cpu, drop_last=True)

dataloader_test = idist.auto_dataloader(ImageDataset(dataroot, transforms_=transforms_, mode='test'), batch_size=batchSize, shuffle=False, num_workers=n_cpu, drop_last=True)

def calc_gen_d2_loss(model, input_fake, input_real):
    # calculate the loss to train D
    outs0 = model.forward(input_fake)
    outs1 = model.forward(input_real)
    loss = 0
    for it, (out0, out1) in enumerate(zip(outs0, outs1)):
        if gan_type == 'lsgan':
            loss += torch.mean((out0 - 1)**2) + torch.mean((out1 - 0)**2)
        elif gan_type == 'nsgan':
            all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
            all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
            loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1) +
                               F.binary_cross_entropy(F.sigmoid(out1), all0))
        else:
            assert 0, "Unsupported GAN type: {}".format(self.gan_type)
    return loss

def calc_dis_loss(model, input_fake, input_real):
        # calculate the loss to train D
        outs0 = model.forward(input_fake)
        outs1 = model.forward(input_real)
        loss = 0
        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

def training_step(engine, batch):
    
    netG_A2B.train()
    netG_B2A.train()
    netD_A.train()
    netD_B.train()
    dis_2.train()
    # Set model input
    real_A = Variable(input_A.copy_(batch['A']))
    real_B = Variable(input_B.copy_(batch['B']))
    ###### Generators A2B and B2A ######
    optimizer_G.zero_grad()

    # Identity loss
    # G_A2B(B) should equal B if real B is fed
    same_B,_,_ = netG_A2B(real_B)
    loss_identity_B = criterion_identity(same_B, real_B)*5.0
    # G_B2A(A) should equal A if real A is fed
    same_A,_,_ = netG_B2A(real_A)
    loss_identity_A = criterion_identity(same_A, real_A)*5.0

    # GAN loss
    fake_B, mask_B, temp_B = netG_A2B(real_A)
    pred_fake = netD_B(fake_B)
    loss_GAN_A2B = criterion_GAN(pred_fake.squeeze(), target_real.squeeze())


    fake_A, mask_A, temp_A = netG_B2A(real_B)
    pred_fake = netD_A(fake_A)
    loss_GAN_B2A = criterion_GAN(pred_fake.squeeze(), target_real.squeeze())

    # Cycle loss
    recovered_A, _, _ = netG_B2A(fake_B)
    loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

    recovered_B, _, _ = netG_A2B(fake_A)
    loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0
    
    x_A_A1_pair = torch.cat((real_A, recovered_A), -3)
    x_A_A2_pair = torch.cat((real_A, same_A), -3)
    x_B_B1_pair = torch.cat((real_B, recovered_B), -3)
    x_B_B2_pair = torch.cat((real_B, same_B), -3)
    loss_dis_2_A = calc_gen_d2_loss(dis_2, x_A_A1_pair, x_A_A2_pair)
    loss_dis_2_B = calc_gen_d2_loss(dis_2, x_B_B1_pair, x_B_B2_pair)
    loss_dis = loss_dis_2_A + loss_dis_2_B
    
    loss_reg_A = lambda_reg_attn * (torch.sum(torch.abs(mask_A[:, :, :, :-1] - mask_A[:, :, :, 1:])) +torch.sum(torch.abs(mask_A[:, :, :-1, :] - mask_A[:, :, 1:, :])))

    loss_reg_B = lambda_reg_attn * (torch.sum(torch.abs(mask_B[:, :, :, :-1] - mask_B[:, :, :, 1:])) + torch.sum(torch.abs(mask_B[:, :, :-1, :] - mask_B[:, :, 1:, :])))
    
    if epoch < gan_curriculum:
        rate = starting_rate
    else:
        rate = default_rate
    
    # Total loss
    loss_G = loss_identity_A + loss_identity_B + (loss_GAN_A2B + loss_GAN_B2A +(loss_reg_A + loss_reg_B)) + loss_dis
    loss_G.backward()
    optimizer_G.step()
    ###################################

    ###### Discriminator A ######
    optimizer_D_A.zero_grad()
    optimizer_D_B.zero_grad()
    optimizer_D_2.zero_grad()
    
    rotated_real_A = random_rotator(real_A).to(idist.device())
    rotated_real_B = random_rotator(real_B).to(idist.device())
    rotated_fake_A = random_rotator(fake_A.detach()).to(idist.device())
    rotated_fake_B = random_rotator(fake_B.detach()).to(idist.device())
    
    loss_ccr_real = (criterion_mse(netD_A(rotated_real_A).squeeze(),netD_A(real_A).squeeze()) + criterion_mse(netD_B(real_B).squeeze(), netD_B(rotated_real_B).squeeze()))*lambda_real
    loss_crr_fake = (criterion_mse(netD_B(rotated_fake_B).squeeze(),netD_B(fake_B.detach()).squeeze()) + criterion_mse(netD_A(rotated_fake_A).squeeze(), netD_A(fake_A.detach()).squeeze()))*lambda_fake
    total_crr_loss = loss_ccr_real + loss_crr_fake
    
    total_crr_loss.backward()

    # Real loss
    pred_real = netD_A(real_A)
    loss_D_real = criterion_GAN(pred_real.squeeze(), target_real.squeeze())

    # Fake loss
    fake_A = fake_A_buffer.push_and_pop(fake_A)
    pred_fake = netD_A(fake_A.detach())
    loss_D_fake = criterion_GAN(pred_fake.squeeze(), target_fake.squeeze())
    # print("386 ",pred_fake.squeeze(), target_fake.squeeze())
    # Total loss
    loss_D_A = (loss_D_real + loss_D_fake)*0.5
    loss_D_A.backward()

    optimizer_D_A.step()
    ###################################

    ###### Discriminator B ######

    # Real loss
    pred_real = netD_B(real_B)
    loss_D_real = criterion_GAN(pred_real.squeeze(), target_real.squeeze())

    # Fake loss
    fake_B = fake_B_buffer.push_and_pop(fake_B)
    pred_fake = netD_B(fake_B.detach())
    loss_D_fake = criterion_GAN(pred_fake.squeeze(), target_fake.squeeze())
    
    # Total loss
    loss_D_B = (loss_D_real + loss_D_fake)*0.5
    loss_D_B.backward()

    optimizer_D_B.step()
    ###################################
    #### Discriminator 2 ####
    x_A_A1_pair = torch.cat((real_A, recovered_A.detach()), -3)
    x_A_A2_pair = torch.cat((real_A, same_A.detach()), -3)
    x_B_B1_pair = torch.cat((real_B, recovered_B.detach()), -3)
    x_B_B2_pair = torch.cat((real_B, same_B.detach()), -3)
    loss_dis_2_A = calc_dis_loss(dis_2, x_A_A1_pair, x_A_A2_pair)
    loss_dis_2_B = calc_dis_loss(dis_2, x_B_B1_pair, x_B_B2_pair)
    loss_dis = loss_dis_2_A + loss_dis_2_B
    
    loss_dis.backward()
    optimizer_D_2.step()
    
    return {
        'loss_G': loss_G.item(),
        'loss_G_identity' : loss_identity_A.item() + loss_identity_B.item(),
        'loss_G_GAN': loss_GAN_A2B.item() + loss_GAN_B2A.item(),
        'loss_G_cycle': loss_cycle_ABA.item() + loss_cycle_BAB.item(),
        'loss_D': loss_D_A.item() + loss_D_B.item() + loss_dis.item(),
        'real_A': real_A,
        'real_B': real_B,
        'fake_A': fake_A,
        'fake_B': fake_B,
        'recovered_A': recovered_A,
        'recovered_B': recovered_B
    }

trainer = Engine(training_step)
@trainer.on(Events.STARTED)
def init_weights():
    if not resume:
        netG_A2B.apply(weights_init_normal)
        netG_B2A.apply(weights_init_normal)
        netD_A.apply(weights_init_normal)
        netD_B.apply(weights_init_normal)
        dis_2.apply(weights_init_normal)
    else:
        x = torch.load(f'output/{epoch}.pth')
        netG_A2B.load_state_dict(x['netG_A2B'])
        netG_B2A.load_state_dict(x['netG_B2A'])
        netD_A.load_state_dict(x['netD_A'])
        netD_B.load_state_dict(x['netD_B'])

G_losses = []
G_losses_identity = []
G_losses_GAN = []
G_losses_cycle = []
D_losses = []


@trainer.on(Events.ITERATION_COMPLETED)
def store_losses(engine):
    o = engine.state.output
    G_losses.append(o["loss_G"])
    D_losses.append(o["loss_D"])
    G_losses_identity.append(o["loss_G_identity"])
    G_losses_GAN.append(o["loss_G_GAN"])
    G_losses_cycle.append(o["loss_G_cycle"])
    
    with open('G_losses.pkl', 'wb') as f:
        pickle.dump(G_losses, f)

    with open('D_losses.pkl', 'wb') as f:
        pickle.dump(D_losses, f)
    
    with open('G_losses_identity.pkl', 'wb') as f:
        pickle.dump(G_losses_identity, f)

    with open('G_losses_GAN.pkl', 'wb') as f:
        pickle.dump(G_losses_GAN, f)
    
    with open('G_losses_cycle.pkl', 'wb') as f:
        pickle.dump(G_losses_cycle, f)

from ignite.metrics import FID, InceptionScore

fid_metric = FID(device=idist.device())
is_metric = InceptionScore(device=idist.device(), output_transform=lambda x: x[0])

fid_metric_1 = FID(device=idist.device())
is_metric_1 = InceptionScore(device=idist.device(), output_transform=lambda x: x[0])

import PIL.Image as Image

def evaluation_step(engine, batch):
    with torch.no_grad():
        netG_B2A.eval()
        input_A = Tensor(batchSize, input_nc, size, size)
        input_B = Tensor(batchSize, output_nc, size, size)
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))
        fake_A = 0.5*(netG_B2A(real_B)[0].data + 1.0)
        recovered_B,_,_ = netG_A2B(fake_A)
        A_output_imgs = []
        A_output_imgs.extend(real_B[:5])
        A_output_imgs.extend(fake_A[:5])
        A_output_imgs.extend(recovered_B[:5])
        save_image(A_output_imgs, 'output/A/%04d.png' % (engine.state.iteration+1))
        return fake_A, real_A

def evaluation_step_1(engine, batch):
    with torch.no_grad():
        netG_A2B.eval()
        input_A = Tensor(batchSize, input_nc, size, size)
        input_B = Tensor(batchSize, output_nc, size, size)
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))
        fake_B = 0.5*(netG_A2B(real_A)[0].data + 1.0)
        recovered_A,_,_ = netG_B2A(fake_B)
        B_output_imgs = []
        B_output_imgs.extend(real_A[:5])
        B_output_imgs.extend(fake_B[:5])
        B_output_imgs.extend(recovered_A[:5])
        save_image(B_output_imgs, 'output/B/%04d.png' % (engine.state.iteration+1))
        return fake_B, real_B

evaluator = Engine(evaluation_step)
evaluator1 = Engine(evaluation_step_1)

fid_metric.attach(evaluator, "fid")
is_metric.attach(evaluator, "is")

fid_metric_1.attach(evaluator1, "fid")
is_metric_1.attach(evaluator1, "is")



fid_values = []
is_values = []

fid_values_1 = []
is_values_1 = []

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    evaluator.run(dataloader_test,max_epochs=1)
    metrics = evaluator.state.metrics
    fid_score = metrics['fid']
    is_score = metrics['is']
    fid_values.append(fid_score)
    is_values.append(is_score)
    
    print(f"Epoch [{engine.state.epoch}] Metric Scores")
    print(f"*   FID : {fid_score:4f}")
    print(f"*    IS : {is_score:4f}")
    
    evaluator1.run(dataloader_test,max_epochs=1)
    metrics = evaluator1.state.metrics
    fid_score = metrics['fid']
    is_score = metrics['is']
    fid_values_1.append(fid_score)
    is_values_1.append(is_score)
    
    print(f"Epoch [{engine.state.epoch}] Metric Scores")
    print(f"*   FID : {fid_score:4f}")
    print(f"*    IS : {is_score:4f}")
    
    with open('fid.pkl', 'wb') as f:
        pickle.dump(fid_values, f)

    with open('is.pkl', 'wb') as f:
        pickle.dump(is_values, f)
    
    with open('fid_1.pkl', 'wb') as f:
        pickle.dump(fid_values_1, f)

    with open('is_1.pkl', 'wb') as f:
        pickle.dump(is_values_1, f)
    
    if engine.state.epoch % 5 == 0:
            weights = {}
            weights['netG_A2B'] = netG_A2B.state_dict()
            weights['netG_B2A'] = netG_B2A.state_dict()
            weights['netD_A'] = netD_A.state_dict()
            weights['netD_B'] = netD_B.state_dict()
            weights['epoch'] = engine.state.epoch
            torch.save(weights, f'output/{engine.state.epoch}.pth')
    
    o = engine.state.output        
    A_output_imgs = []
    A_output_imgs.extend(o['real_A'])
    A_output_imgs.extend(o['fake_B'])
    A_output_imgs.extend(o['recovered_A'])
    B_output_imgs = []
    B_output_imgs.extend(o['real_B'])
    B_output_imgs.extend(o['fake_A'])
    B_output_imgs.extend(o['recovered_B'])
    save_image(A_output_imgs, f'images_A_{engine.state.epoch}.png')
    save_image(B_output_imgs, f'images_B_{engine.state.epoch}.png')

from ignite.metrics import RunningAverage


RunningAverage(output_transform=lambda x: x["loss_G"]).attach(trainer, 'loss_G')
RunningAverage(output_transform=lambda x: x["loss_D"]).attach(trainer, 'loss_D')

from ignite.contrib.handlers import ProgressBar

ProgressBar().attach(trainer, metric_names=['loss_G','loss_D'])
ProgressBar().attach(evaluator)
ProgressBar().attach(evaluator1)

# def training(*args):
trainer.run(dataloader, max_epochs=50)

# with idist.Parallel(backend='nccl') as parallel:
#     parallel.run(training)