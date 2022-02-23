import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, nc, nf=48):
        super(Generator, self).__init__()
        self.nf = nf

        self.input_layer = nn.Linear(100, nf*8)
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(in_channels=nf*8, out_channels=nf*4, kernel_size=4, stride=2, padding=0, bias=False),
                                    nn.BatchNorm2d(nf*4),
                                    nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(in_channels=nf*4, out_channels=nf*2, kernel_size=4, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(nf*2),
                                    nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(in_channels=nf*2, out_channels=nf*1, kernel_size=4, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(nf*1),
                                    nn.ReLU(True))
        self.output_layer = nn.Sequential(nn.ConvTranspose2d(in_channels=nf*1, out_channels=nc, kernel_size=4, stride=2, padding=1, bias=False),
                                    nn.Tanh())

    def forward(self, x):
        out = self.input_layer(x)
        out = self.layer1(out.view(-1, self.nf*8, 1, 1))
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.output_layer(out)
        return out



class Discriminator(nn.Module):
    def __init__(self, nc, nf, num_classes):
        super(Discriminator, self).__init__()
        self.nf = nf

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=nc, out_channels=nf*1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf*1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=nf, out_channels=nf*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=nf*2, out_channels=nf*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=nf*4, out_channels=nf*8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(nf*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=nf*8, out_channels=nf*16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf*16),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.output_layer = nn.Sequential(
            nn.Conv2d(in_channels=nf*16, out_channels=nf*32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(nf*32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.avd_layer = nn.Sequential(nn.Linear(4 * 4 * nf*32, 1),
                                       nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(4 * 4 * nf*32, num_classes),
                                       nn.Softmax())

    def forward(self, x):
        out = self.input_layer(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.output_layer(out)
        out = out.view(-1, 4 * 4 * self.nf*32)
        avd_out = self.avd_layer(out)
        aux_out = self.aux_layer(out)

        return avd_out, aux_out

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
    batch_size = 128
    imb_factor = 0.01
    epochs = 200
    learning_rate = 0.0002
    beta1 = 0.5
    beta2 = 0.999



    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)




    # Define dataset
    # train_loader = ImbalanceMNISTDataLoader(data_dir="../../data", batch_size=batch_size, shuffle=True, num_workers=1, training=True, balanced=False,
    #              retain_epoch_size=True, imb_factor=imb_factor)
    #
    # test_loader = ImbalanceMNISTDataLoader(data_dir="../../data", batch_size=batch_size, shuffle=False, num_workers=1, training=False, balanced=False, retain_epoch_size=True)

    train_loader = ImbalanceCIFAR10DataLoader(data_dir="../../data", batch_size=batch_size, shuffle=True, num_workers=1, training=True, balanced=False,
                 retain_epoch_size=True, imb_factor=imb_factor)

    test_loader = ImbalanceCIFAR10DataLoader(data_dir="../../data", batch_size=batch_size, shuffle=False, num_workers=1, training=False, balanced=False, retain_epoch_size=True)


    print(len(train_loader))
    print(len(test_loader))

    print(train_loader.cls_num_list)
    print(test_loader.cls_num_list)

    img, target = iter(train_loader).__next__()
    print(img.size())

    img_grid = make_grid(img, normalize=True)
    plt.imshow(img_grid.permute(1,2,0))
    plt.show()



    # Define models
    G = Generator(nc = 3, nf=96).to(device)
    D = Discriminator(nc = 3, nf=32, num_classes=10).to(device)
    # G.apply(weights_init)
    # D.apply(weights_init)
    print(G)
    print(D)


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

            input_noise = torch.randn((_batch, 100)).to(device)
            real_avd = torch.ones(_batch, 1).to(device)
            fake_avd = torch.zeros(_batch, 1).to(device)

            # Training Discriminator
            optimizer_D.zero_grad()

            out_avd, out_cls = D(img)
            avd_loss = avd_criterion(out_avd, real_avd)
            cls_loss = cls_criterion(out_cls, target)
            d_real_loss = avd_loss + cls_loss
            d_real_loss.backward()
            D_x_score = avd_loss.data.mean()

            pred = out_cls.argmax(1)
            correct = (pred == target).cpu().sum()
            acc = correct / _batch


            gened_img = G(input_noise)
            out_avd, out_cls = D(gened_img.detach())
            avd_loss = avd_criterion(out_avd, fake_avd)
            cls_loss = cls_criterion(out_cls, target)
            d_fake_loss = avd_loss + cls_loss
            d_fake_loss.backward()

            D_G_socre = avd_loss.data.mean()
            optimizer_D.step()

            d_loss = d_real_loss + d_fake_loss

            # Training Generator
            G.zero_grad()
            out_avd, out_cls = D(gened_img)
            avd_loss = avd_criterion(out_avd, real_avd)
            cls_loss = cls_criterion(out_cls, target)

            g_loss = avd_loss + cls_loss
            g_loss.backward()

            G_score = avd_loss.data.mean()
            optimizer_G.step()


            # compute the average loss
            curr_iter = epoch * len(train_loader) + i
            all_loss_G = avg_loss_G * curr_iter
            all_loss_D = avg_loss_D * curr_iter
            all_loss_A = avg_loss_A * curr_iter
            all_loss_G += g_loss.item()
            all_loss_D += d_loss.item()
            all_loss_A += acc
            avg_loss_G = all_loss_G / (curr_iter + 1)
            avg_loss_D = all_loss_D / (curr_iter + 1)
            avg_loss_A = all_loss_A / (curr_iter + 1)


        print('[%d/%d][%d/%d] Loss_D: %.4f (%.4f) Loss_G: %.4f (%.4f) D(x): %.4f D(G(z)): %.4f / %.4f Acc: %.4f (%.4f)' % (epoch, epochs, i, len(train_loader), d_loss.item(), avg_loss_D, g_loss.item(), avg_loss_G, D_x_score, D_G_socre, G_score, acc, avg_loss_A))


        img_grid = make_grid(gened_img.detach().cpu(), normalize=True)
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


