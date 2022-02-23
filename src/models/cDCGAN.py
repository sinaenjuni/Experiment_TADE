import torch
import torch.nn as nn
import torch.nn.functional as F

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator
class Generator(nn.Module):
    def __init__(self, nz, nc, ncls, ngf):
        super(Generator, self).__init__()

        self.noise = nn.Sequential(
            nn.ConvTranspose2d(nz, (ngf * 4)//2, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d((ngf * 4)//2),
            nn.LeakyReLU(0.2, True)
        )

        self.condition = nn.Sequential(
            nn.ConvTranspose2d(ncls, (ngf * 4)//2, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d((ngf * 4)//2),
            nn.LeakyReLU(0.2, True)
        )

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(ngf * 8),
            # nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            # nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(ngf * 4),
            # nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, noise, condition):
        noise = self.noise(noise)
        condition = self.condition(condition)
        concat = torch.cat([noise, condition], 1)
        out = self.main(concat)
        return out
        # return z


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            # nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            # nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False)
            # nn.Sigmoid()
        )
        self.adv = nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.cls = nn.Conv2d(ndf * 4, 10, kernel_size=4, stride=1, padding=0, bias=False)

    def forward(self, input):
        y = self.main(input)
        adv = self.adv(y)
        cls = self.cls(y)
        return adv, cls
        # return self.main(input)




if __name__ == '__main__':
    import torch
    import torch.nn.functional as F

    nc = 3
    ndf = 32

    nz = 100
    ngf = 32
    ncls = 10
    batch_size = 128

# fixed data for generator input
    g_fixed_label = torch.zeros(ncls*10, ncls)
    g_fixed_label_index = torch.LongTensor([ i//10 for i in range(100) ]).view(ncls*10, 1)
    g_fixed_label = g_fixed_label.scatter_(1, g_fixed_label_index, 1).view(ncls*10, ncls, 1, 1)

    g_fixed_noise = torch.randn((10, 100, 1, 1)).repeat(10, 1, 1, 1)
    print(g_fixed_noise.size())
    print(g_fixed_label.size())

# label format for training generator
    g_label = torch.zeros((ncls, ncls))
    g_label_index = torch.LongTensor([ i for i in range(10) ]).view(ncls, 1)
    g_label = g_label.scatter_(1, g_label_index, 1).view(ncls, ncls, 1, 1)

# define model
    # print(onehot)
    G = Generator(nz=nz, nc=nc, ngf=ngf, ncls=ncls)
    D = Discriminator(nc=nc, ndf=ndf)

    for g in G.named_parameters():
        print(g)

    # noise = torch.randn((batch_size, nz, 1, 1))
    # condition = torch.rand((batch_size, ncls, 1, 1))
    # print(noise.size(), condition.size())
    g_out = G(g_fixed_noise, g_fixed_label)
    print(g_out.size())

    img = torch.randn((100, nc, 32, 32))
    d_out_adv, d_out_cls = D(img)
    d_out_adv = d_out_adv.squeeze()
    d_out_cls = d_out_cls.squeeze()
    print(d_out_adv.size(), d_out_cls.size())

    fixed_label = torch.LongTensor([i//10 for i in range(100)]).view(100)
    print(fixed_label.size())
    loss = F.cross_entropy(d_out_cls, fixed_label)
    print(loss)
    # img_size = 32
    #
    # c_ = (torch.rand(batch_size, 1) * ncls).long().squeeze()
    # # cls = onehot[c_]
    # cls = onehot[c_]
    #
    # img = torch.randn((batch_size, nc, img_size, img_size))
    # labels = torch.randint(10, (batch_size,))
    # z = torch.randn((batch_size, nz, 1, 1))
    # print(z.dtype)
    # print(img.size())
    # print(z.size())
    # d_out_adv, d_out_cls = D(img)
    # g_out = G(z, cls)
    #
    # print(d_out_adv.size(), d_out_cls.size())
    # print(g_out.size())
    #
    # d_out_adv = d_out_adv.view(-1)
    # d_out_cls = d_out_cls.view(-1, 10)
    # print(d_out_adv.size(), d_out_cls.size())
    #
    # cls_loss = F.cross_entropy(d_out_cls, labels)
    # print(cls_loss)
