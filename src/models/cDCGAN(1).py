import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def getCBR(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels,
                                                out_channels=out_channels,
                                                kernel_size=kernel_size,
                                                stride=stride,
                                                padding=padding,
                                                bias=False),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(True))
    def __init__(self, nc, nf=48):
        super(Generator, self).__init__()
        self.nf = nf

        self.input_layer = self.getCBR(100, nf*2, kernel_size=4, stride=1, padding=0)

        self.condition_layer = self.getCBR(10, nf*2, kernel_size=4, stride=1, padding=0)

        self.layer1 = self.getCBR(nf*2*2, nf*2, kernel_size=4, stride=2, padding=1)

        self.layer2 = self.getCBR(nf*2, nf*1, kernel_size=4, stride=2, padding=1)

        self.output_layer = nn.Sequential(nn.ConvTranspose2d(in_channels=nf*1,
                                                             out_channels=nc,
                                                             kernel_size=4,
                                                             stride=2,
                                                             padding=1,
                                                             bias=False),
                                            nn.Tanh())
    def forward(self, x, c):
        x = self.input_layer(x)
        c = self.condition_layer(c)
        out = torch.cat([x, c], 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.output_layer(out)
        return out



class Discriminator(nn.Module):
    def getCBL(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        return  nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def getCL(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        return  nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def __init__(self, nc, nf, num_classes):
        super(Discriminator, self).__init__()
        self.nf = nf

        self.input_layer = self.getCL(nc, nf*1, 3, 2, 1, True)

        self.condition_layer = self.getCL(num_classes, nf*1, 3, 2, 1, True)

        self.layer1 = self.getCBL(nf*1*2, nf*4, 3, 1, 1, False)

        self.layer2 = self.getCBL(nf*4, nf*8, 3, 2, 1, False)

        self.layer3 = self.getCBL(nf*8, nf*16, 3, 1, 1, False)

        self.layer4 = self.getCBL(nf*16, nf*32, 3, 2, 1, False)

        self.output_layer = nn.Sequential(nn.Linear(4 * 4 * nf*32, 1),
                                       nn.Sigmoid())

    def forward(self, x, c):
        x = self.input_layer(x)
        c = self.condition_layer(c)
        out = torch.cat([x, c], 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(-1, 4 * 4 * self.nf*32)
        out = self.output_layer(out)
        return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

if __name__ == "__main__":
    from utiles.imbalance_mnist_loader import ImbalanceMNISTDataLoader
    from utiles.imbalance_cifar10_loader import ImbalanceCIFAR10DataLoader
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt


    # Hyper-parameters
    batch_size = 32
    imb_factor = 0.01
    epochs = 200
    learning_rate = 0.0002
    beta1 = 0.5
    beta2 = 0.999



    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)



    # Define dataset
    train_loader = ImbalanceMNISTDataLoader(data_dir="../../data", batch_size=batch_size, shuffle=True, num_workers=1, training=True, balanced=False,
                 retain_epoch_size=True, imb_factor=0.01)

    test_loader = ImbalanceMNISTDataLoader(data_dir="../../data", batch_size=batch_size, shuffle=False, num_workers=1, training=False, balanced=False, retain_epoch_size=True)

    # train_loader = ImbalanceCIFAR10DataLoader(data_dir="../../data", batch_size=batch_size, shuffle=True, num_workers=1, training=True, balanced=False,
    #              retain_epoch_size=True, imb_factor=imb_factor)
    #
    # test_loader = ImbalanceCIFAR10DataLoader(data_dir="../../data", batch_size=batch_size, shuffle=False, num_workers=1, training=False, balanced=False, retain_epoch_size=True)

    print(len(train_loader))
    print(len(test_loader))

    print(train_loader.cls_num_list)
    print(test_loader.cls_num_list)

    img, target = iter(train_loader).__next__()
    print(img.size())

    img_grid = make_grid(img, normalize=True)
    plt.imshow(img_grid.permute(1,2,0))
    plt.show()

    index = torch.tensor([[i // 10] for i in range(100)])
    fixed_condition = torch.zeros(100, 10).scatter_(1, index, 1).to(device)
    fixed_condition = fixed_condition.view(100,10,1,1)

    print(fixed_condition)
    fixed_noise = torch.randn(10, 100, 1, 1).repeat(10, 1, 1, 1).to(device)
    print(fixed_noise)


    onehot = torch.eye(10).to(device)

    fill = torch.zeros([10, 10, 32, 32]).to(device)
    for i in range(10):
        fill[i, i, :, :] = 1


    # Define models
    G = Generator(nc = 1, nf=12).to(device)
    D = Discriminator(nc = 1, nf=16, num_classes=10).to(device)
    G.apply(weights_init)
    D.apply(weights_init)
    # print(G)
    # print(D)
    # rand_label = (torch.rand(100)*10).type(torch.long)
    # print(rand_label)
    # g_out = G(fixed_noise.view(-1, 100, 1, 1), fixed_condition.view(-1, 10, 1, 1))
    # print(g_out.size())
    # d_out = D(g_out, fill[rand_label])
    # print(d_out.size())


    # input_noise = torch.randn((128, 100)).to(device)
    # gened_img = G(input_noise, onehot[target])
    # print(gened_img.size())


    # Set optimizer
    optimizer_G = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(beta1, beta2))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(beta1, beta2))


    # Set criterion
    avd_criterion = nn.BCELoss()
    cls_criterion = nn.NLLLoss()

    avg_loss_D = 0.0
    avg_loss_G = 0.0
    avg_loss_A = 0.0

    # Train
    for epoch in range(epochs):
        for i, (img, target) in enumerate(train_loader):
            img = img.to(device)
            target = target.to(device)

            _batch = img.size(0)

            input_noise = torch.randn((_batch, 100, 1, 1)).to(device)
            real_avd = torch.ones(_batch, 1).to(device)
            fake_avd = torch.zeros(_batch, 1).to(device)

            # Training Discriminator
            optimizer_D.zero_grad()

            out_avd = D(img, fill[target])
            avd_loss = avd_criterion(out_avd, real_avd)
            # cls_loss = cls_criterion(out_cls, target)
            # d_real_loss = avd_loss + cls_loss
            d_real_loss = avd_loss
            d_real_loss.backward()
            D_x_score = avd_loss.data.mean()

            # pred = out_cls.argmax(1)
            # correct = (pred == target).cpu().sum()
            # acc = correct / _batch


            gened_img = G(input_noise, onehot[target].view(-1, 10, 1, 1))
            # out_avd, out_cls = D(gened_img.detach(), fill[target])
            out_avd = D(gened_img.detach(), fill[target])
            avd_loss = avd_criterion(out_avd, fake_avd)
            # cls_loss = cls_criterion(out_cls, target)
            # d_fake_loss = avd_loss + cls_loss
            d_fake_loss = avd_loss
            d_fake_loss.backward()

            D_G_socre = avd_loss.data.mean()
            optimizer_D.step()

            d_loss = d_real_loss + d_fake_loss

            # Training Generator
            G.zero_grad()
            # out_avd, out_cls = D(gened_img, fill[target])
            out_avd = D(gened_img, fill[target])
            avd_loss = avd_criterion(out_avd, real_avd)
            # cls_loss = cls_criterion(out_cls, target)

            # g_loss = avd_loss + cls_loss
            g_loss = avd_loss
            g_loss.backward()

            G_score = avd_loss.data.mean()
            optimizer_G.step()


            # compute the average loss
            curr_iter = epoch * len(train_loader) + i
            all_loss_G = avg_loss_G * curr_iter
            all_loss_D = avg_loss_D * curr_iter
            # all_loss_A = avg_loss_A * curr_iter
            all_loss_G += g_loss.item()
            all_loss_D += d_loss.item()
            # all_loss_A += acc
            avg_loss_G = all_loss_G / (curr_iter + 1)
            avg_loss_D = all_loss_D / (curr_iter + 1)
            # avg_loss_A = all_loss_A / (curr_iter + 1)


        print('[%d/%d][%d/%d] Loss_D: %.4f (%.4f) Loss_G: %.4f (%.4f) D(x): %.4f D(G(z)): %.4f / %.4f' % (epoch, epochs, i, len(train_loader), d_loss.item(), avg_loss_D, g_loss.item(), avg_loss_G, D_x_score, D_G_socre, G_score))

        test_img = G(fixed_noise, fixed_condition)
        # img_grid = make_grid(gened_img.detach().cpu(), normalize=True)
        img_grid = make_grid(test_img.detach().cpu(), normalize=True, nrow=10)
        plt.imshow(img_grid.permute(1,2,0))
        plt.show()

    #
    # input_noise = torch.randn((32, 110))
    # input_tensor = torch.rand((32, 3, 32, 32))
    #
    #
    # gened_images = G(input_noise)
    # print(gened_images.size())
    #
    # avd_out, aux_out = D(gened_images)
    # print(avd_out.size(), aux_out.size())
    #
    #


