import os

import numpy as np
import torch
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns

from torchsummaryX import summary

from utiles.tensorboard import getTensorboard
from utiles.data import getSubDataset
from utiles.imbalance_cifar10_loader import ImbalanceCIFAR10DataLoader
from models.resnet_s_D import resnet32
import models.DCGAN_scaleup as Generator

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)



# Define hyper-parameters
name = 'LSGAN/cifar10/aug/p100'
TENSORBOARD_PATH = f'../../tb_logs/{name}/'

# Save the model checkpoints
SAVE_PATH = f'../../weights/{name}/'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


num_workers = 4
num_epochs = 200
batch_size = 128
imb_factor = 0.01

learning_rate = 0.0002
weight_decay = 5e-4
momentum = 0.9
nesterov = True

nz = 100
nc = 3
ngf = 64
beta1 = 0.5
beta2 = 0.999
lambda_gp = 10
n_critic = 3

fixed_noise = torch.randn((100, nz, 1, 1)).to(device)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Define Tensorboard
tb = getTensorboard(TENSORBOARD_PATH)



# Define DataLoader
train_data_loader = ImbalanceCIFAR10DataLoader(data_dir='../../data',
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              training=True,
                                              imb_factor=imb_factor)

test_data_loader = ImbalanceCIFAR10DataLoader(data_dir='../../data',
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              training=False)

print("Number of train dataset", len(train_data_loader.dataset))
print("Number of test dataset", len(test_data_loader.dataset))

print(train_data_loader.cls_num_list)
cls_num_list = train_data_loader.cls_num_list


# # Generator
# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.main = nn.Sequential(
#             # input is Z, going into a convolution
#             # nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
#             # nn.BatchNorm2d(ngf * 8),
#             # nn.ReLU(True),
#             # state size. (ngf*8) x 4 x 4
#             nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#             # state size. (ngf*4) x 8 x 8
#             nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#             # state size. (ngf*2) x 16 x 16
#             nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True),
#             # state size. (ngf) x 32 x 32
#             nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
#             nn.Tanh()
#             # state size. (nc) x 64 x 64
#         )
#     def forward(self, input):
#         return self.main(input)


# Define model
G = Generator.generator().to(device)
# G = Generator().to(device)
D = resnet32(num_classes=10, use_norm=True).to(device)

G.apply(weights_init)
D.apply(weights_init)


summary(G, torch.rand(32, 100, 1, 1).to(device))
summary(D, torch.rand(32, 3, 32, 32).to(device))


# Define optimizer
d_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(beta1, beta2))
g_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(beta1, beta2))



# Training model
total_step = len(train_data_loader)
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_data_loader):
        # images = images.reshape(batch_size, -1).to(device)
        batch = images.size(0)
        images = images.to(device)

        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones((batch,) ).to(device)
        fake_labels = torch.zeros((batch,) ).to(device)

        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #
        z = torch.randn(batch, nz, 1, 1).to(device)

        g_optimizer.zero_grad()

        fake_images = G(z)
        g_outputs = D(fake_images).view(batch, -1)

        g_loss = torch.mean((g_outputs - real_labels) ** 2)
        score_G = g_outputs.mean().item()

        g_loss.backward()
        g_optimizer.step()

        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #
        d_optimizer.zero_grad()

        real_output = D(images).view(batch, -1)
        d_real_loss = 0.5 * torch.mean((real_output - real_labels) ** 2)
        d_real_score = real_output.mean().item()

        z = torch.randn(batch, nz, 1, 1).to(device) # mean==0, std==1
        fake_images = G(z)

        fake_output = D(fake_images.detach()).view(batch, -1)
        d_fake_loss = 0.5 * torch.mean(fake_output - fake_labels ** 2)
        d_fake_score = fake_output.mean().item()

        # d_loss = loss_D_real + loss_D_fake + lambda_gp * gradient_penalty
        d_loss = 0.5 * (d_real_loss + d_fake_loss)

        # reset_grad()
        d_loss.backward()
        d_optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f} / {:.2f}'
                  .format(epoch+1, num_epochs, i + 1, total_step,
                          d_loss.item(), g_loss.item(),
                          d_real_score, d_fake_score, score_G))

    # Save real images
    # if (epoch + 1) == 1:
    #     images = images.reshape(images.size(0), 1, 28, 28)
    #     save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))

    tb.add_scalars(global_step=epoch + 1,
                   main_tag='loss',
                   tag_scalar_dict={'discriminator':d_loss.item(),
                                    'generator':g_loss.item()})

    tb.add_scalars(global_step=epoch + 1,
                   main_tag='score',
                   tag_scalar_dict={'real':d_real_score,
                                    'fake':d_fake_score,
                                    'g':score_G})

    # tb.add_scalar(tag='d_loss', global_step=epoch+1, scalar_value=d_loss.item())
    # tb.add_scalar(tag='g_loss', global_step=epoch+1, scalar_value=g_loss.item())
    # tb.add_scalar(tag='real_score', global_step=epoch+1, scalar_value=real_score.mean().item())
    # tb.add_scalar(tag='fake_score', global_step=epoch+1, scalar_value=fake_score.mean().item())

    with torch.no_grad():
        result_images = make_grid(G(fixed_noise).cpu(), padding=0, nrow=10, normalize=True)
        plt.imshow(result_images.permute(1,2,0).numpy())
        plt.tight_layout()
        plt.show()
        tb.add_image(tag='gened_images',
                      global_step=epoch+1,
                      img_tensor=result_images)


    # Save sampled images
    # fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    # save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch + 1)))



    torch.save(G.state_dict(), SAVE_PATH + f'G_{epoch+1}.pth')
    torch.save(D.state_dict(), SAVE_PATH + f'D_{epoch+1}.pth')









