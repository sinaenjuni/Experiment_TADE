import torch.nn as nn

# From https://github.com/kaidic/LDAM-DRW/blob/master/models/resnet_cifar.py
'''
Properly implemented ResNet for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(BasicBlock, self).__init__()
        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.ConvTranspose2d(in_planes, planes, kernel_size=4, stride=2, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.planes = planes
                self.in_planes = in_planes
                # self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2],
                                                  (0, 0, 0, 0, (planes - in_planes) // 2, (planes - in_planes) // 2),
                                                  "constant", 0))

            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.ConvTranspose2d(in_planes, self.expansion * planes, kernel_size=4, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), inplace=True, negative_slope=0.2)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out, inplace=True, negative_slope=0.2)
        return out


class Generator(nn.Module):
    def __init__(self, block, num_blocks, reduce_dimension=False, layer2_output_dim=None,
                 layer3_output_dim=None, s=30):
        super(Generator, self).__init__()
        self.in_planes = 256

        self.conv1 = nn.ConvTranspose2d(100, 256, kernel_size=4, stride=2, padding=0, bias=False)  # 4
        self.bn1 = nn.BatchNorm2d(256)
        self.layer1 = self._make_layer(block, 256, num_blocks[0], stride=1)

        if layer2_output_dim is None:
            if reduce_dimension:
                layer2_output_dim = 48
            else:
                layer2_output_dim = 128

        if layer3_output_dim is None:
            if reduce_dimension:
                layer3_output_dim = 24
            else:
                layer3_output_dim = 64

        self.layer2 = self._make_layer(block, layer2_output_dim, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, layer3_output_dim, num_blocks[2], stride=2)

        self.last = nn.Sequential(nn.ConvTranspose2d(layer3_output_dim, layer3_output_dim, kernel_size=4, stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(layer3_output_dim),
                                  nn.LeakyReLU(inplace=True, negative_slope=0.2),
                                  nn.Conv2d(layer3_output_dim, 3, kernel_size=3, stride=1, padding=1),
                                  nn.Tanh())

        # if use_norm:
        #     self.linear = NormedLinear(layer3_output_dim, num_classes)
        # else:
        #     s = 1
        #     self.linear = nn.Linear(layer3_output_dim, num_classes)

        self.s = s

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def _hook_before_iter(self):
        assert self.training, "_hook_before_iter should be called at training time only, after train() is called"
        count = 0
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if module.weight.requires_grad == False:
                    module.eval()
                    count += 1

        if count > 0:
            print("Warning: detected at least one frozen BN, set them to eval state. Count:", count)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), inplace=True, negative_slope=0.2)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.last(out)
        # self.feat_before_GAP = out
        # out = F.avg_pool2d(out, out.size()[3])
        # out = out.view(out.size(0), -1)
        # self.feat = out

        # out = self.linear(out)
        # out = out * self.s  # This hyperparam s is originally in the loss function, but we moved it here to prevent using s multiple times in distillation.
        return out

# def test(net):
#     import numpy as np
#     total_params = 0
#
#     for x in filter(lambda p: p.requires_grad, net.parameters()):
#         total_params += np.prod(x.data.numpy().shape)
#     print("Total number of params", total_params)
#     print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))


# if __name__ == "__main__":
#     for net_name in __all__:
#         if net_name.startswith('resnet'):
#             print(net_name)
#             test(globals()[net_name]())
#             print()

def generator():
    return Generator(BasicBlock, [3, 3, 3], reduce_dimension=False)

if __name__ == "__main__":
    from torchsummaryX import summary

    # model = Generator(BasicBlock, [5, 5, 5], reduce_dimension=False)
    model = generator()
    # rest32 = resnet32(10, True)

    summary(model, torch.zeros((32, 100, 1, 1)))

    # out = i(input)
    # print(out.size())
    # input = out
    # output = rest32(input)
    # print(output.size())



# Discriminator
# class Discriminator(nn.Module):
#     def __init__(self, nc, ndf, ngpu):
#         super(Discriminator, self).__init__()
#         self.ngpu = ngpu
#         self.main = nn.Sequential(
#             # input is (nc) x 64 x 64
#             # nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#             # nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 32 x 32
#             nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(ndf),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) x 16 x 16
#             nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*4) x 8 x 8
#             nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*8) x 4 x 4
#             nn.Conv2d(in_channels=ndf * 4, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False)
#             # nn.Sigmoid()
#         )
#
#     def forward(self, input):
#         return self.main(input)
#
# # Generator
# class Generator(nn.Module):
#     def __init__(self, nz, nc, ngf):
#         super(Generator, self).__init__()
#         self.main = nn.Sequential(
#             # input is Z, going into a convolution
#             # nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
#             # nn.BatchNorm2d(ngf * 8),
#             # nn.ReLU(True),
#             # state size. (ngf*8) x 4 x 4
#             nn.ConvTranspose2d(in_channels=nz, out_channels=ngf * 4, kernel_size=4, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#
#             nn.ConvTranspose2d(in_channels=ngf * 4, out_channels=ngf * 4, kernel_size=4, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#
#
#
#         # state size. (ngf*4) x 8 x 8
#             nn.ConvTranspose2d(in_channels=ngf * 4, out_channels=ngf * 2, kernel_size=4, padding=2, stride=1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#
#             nn.ConvTranspose2d(in_channels=ngf * 2, out_channels=ngf * 2, kernel_size=4, padding=2, stride=1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#
#         # state size. (ngf*2) x 16 x 16
#             nn.ConvTranspose2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True),
#
#             nn.ConvTranspose2d(in_channels=ngf, out_channels=ngf, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True),
#
#
#         # state size. (ngf) x 32 x 32
#             nn.ConvTranspose2d(in_channels=ngf, out_channels=nc, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.Tanh()
#             # state size. (nc) x 64 x 64
#         )
#
#     def forward(self, input):
#         return self.main(input)
#
#
# if __name__ == '__main__':
#     import torch
#     nc = 3
#     ndf = 32
#
#     nz = 100
#     ngf = 32
#
#     D = Discriminator(nc, ndf, ngpu=1)
#     G = Generator(nz, ngf, ngpu=1)
#
#     batch_size = 64
#     img_size = 32
#
#     img = torch.randn((batch_size, nc, img_size, img_size))
#     z = torch.randn((batch_size, nz, 1, 1))
#
#     print(img.size())
#     print(z.size())
#     d_out = D(img)
#     g_out = G(z)
#
#     print(d_out.size())
#     print(g_out.size())
#
#     d_out = d_out.view(-1)
#     print(d_out.size())
