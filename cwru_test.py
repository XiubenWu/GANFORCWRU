import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn

from MyModule import ReWrite
from MyModule import G_D_Module
from MyModule import TrainFunction

import dataToCoe

os.makedirs("images", exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--data_length", type=int, default=1024, help="size of the data length")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=20, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--Gchannels", type=int, default=128, help="start_channels_for_G")
parser.add_argument("--n_classes", type=int, default=5, help="num of class of data (labels)")
opt = parser.parse_args()
print(opt)

source_files = os.listdir('coedatas')

cuda = True if torch.cuda.is_available() else False

# cuda = False
img_shape = (opt.channels, opt.img_size, opt.img_size)

plt.rcParams['figure.figsize'] = (opt.img_size, opt.img_size)  # 设置figure_size尺寸
# plt.rcParams['image.interpolation'] = 'nearest'  # 设置 interpolation style
# plt.rcParams['image.cmap'] = 'gray'  # 设置 颜色 style
plt.rcParams['figure.dpi'] = 10


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.data_length, 100),
            nn.BatchNorm1d(100, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(100, 128),
            nn.BatchNorm1d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, opt.data_length)
        )

    def forward(self, z):
        gen_data = self.model(z)
        return gen_data


class GeneratorFor2D(nn.Module):
    def __init__(self):
        super(GeneratorFor2D, self).__init__()

        self.init_size = opt.img_size // 4  # upsample * 2
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, opt.Gchannels * self.init_size ** 2))

        self.channel1 = 32
        self.channel2 = 64
        self.channel3 = 32

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(opt.Gchannels),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(opt.Gchannels, self.channel1, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.channel1, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(self.channel1, self.channel2, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.channel2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.channel2, self.channel3, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.channel3, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel3, opt.channels, 3, stride=1, padding=1),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], opt.Gchannels, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.data_length, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        results = self.model(z)
        return results


class DiscriminatorFor2D(nn.Module):
    def __init__(self):
        super(DiscriminatorFor2D, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(opt.channels, 32, 3, 1, 0),  # 32* to 30*
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(2),  # 30* to 15*

            nn.Conv2d(32, 128, 4, 1, 0),  # 15* to 12*
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(2),  # 12* to 6*

        )
        self.l2 = nn.Sequential(
            nn.Linear(6 * 6 * 128, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        out = self.block(img)
        out = out.view(out.shape[0], -1)
        validity = self.l2(out)

        return validity


def get_data(dataset):
    length = len(dataset)
    data_count = length // opt.data_length
    dataset = dataset[:data_count * opt.data_length]
    dataset = np.reshape(dataset, (data_count, opt.data_length))

    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
    )

    return dataloader


# train Gan
# batch_size = opt.batch_size
# for i, data in enumerate(dataloader):
#     if i % batch_size == 0:
#         batch_data = data.copy()
#     else:
#         torch.cat(batch_data, data)
#         if (i + 1) % batch_size == 0:
#             print(batch_data)
#             print(len(batch_data[0]), len(batch_data[1]))
#             batch_data = data.clone()


def train_gan_for_2d():
    filename = 'coedatas/1.npz'
    dataset = dataToCoe.data_read(filename)
    data_loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
    )
    generator = GeneratorFor2D()
    discriminator = DiscriminatorFor2D()
    loss = nn.BCELoss()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        loss.cuda()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    for epoch in range(opt.n_epochs):
        for i, imgs in enumerate(data_loader):

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

            # Configure input
            imgs = imgs[:, np.newaxis, :, :]
            real_imgs = Variable(imgs.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = loss(discriminator(real_imgs), valid)
            fake_loss = loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(data_loader), d_loss.item(), g_loss.item())
            )

            if epoch % 5 == 0 and i == 1:

                if cuda:
                    gen_imgs = gen_imgs.cpu()
                plt.contourf(gen_imgs[0][0].detach().numpy())
                plt.savefig("waveImages\\" + str(epoch) + '.jpg')
                plt.close()

    torch.save(generator.state_dict(), 'GANParameters\\generator.pt')
    torch.save(discriminator.state_dict(), 'GANParameters\\discriminator.pt')


def train_again_for_2D():
    filename = 'coedatas/0.npz'
    dataset = dataToCoe.data_read(filename)
    data_loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
    )
    generator = GeneratorFor2D()
    discriminator = DiscriminatorFor2D()

    generator.load_state_dict(torch.load('GANParameters\\generator.pt'))
    discriminator.load_state_dict(torch.load('GANParameters\\discriminator.pt'))
    loss = nn.BCELoss()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        loss.cuda()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    for epoch in range(opt.n_epochs):
        for i, imgs in enumerate(data_loader):

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

            # Configure input
            imgs = imgs[:, np.newaxis, :, :]
            real_imgs = Variable(imgs.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = loss(discriminator(real_imgs), valid)
            fake_loss = loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(data_loader), d_loss.item(), g_loss.item())
            )

            if epoch % 5 == 0 and i == 1:

                if cuda:
                    gen_imgs = gen_imgs.cpu()
                plt.contourf(gen_imgs[0][0].detach().numpy())
                plt.axis('off')
                plt.savefig("waveImages\\" + str(epoch + 200) + '.jpg')
                plt.close()

    torch.save(generator.state_dict(), 'GANParameters\\generator.pt')
    torch.save(discriminator.state_dict(), 'GANParameters\\discriminator.pt')


def train_gan():
    datasets = ReWrite.load_data_in_seq(source_files)
    datasets = ReWrite.MyDataSet(datasets)
    data_loader = DataLoader(
        datasets,
        batch_size=16,
        shuffle=True,
    )
    generator = Generator()
    discriminator = Discriminator()
    loss = nn.BCELoss()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        loss.cuda()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    for epoch in range(opt.n_epochs):
        for i, data in enumerate(data_loader):
            valid = Variable(Tensor(data.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(data.size(0), 1).fill_(0.0), requires_grad=False)

            real_data = Variable(data.type(Tensor))

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (data.size(0), opt.data_length))))

            gen_data = generator(z)

            g_loss = loss(discriminator(gen_data), valid)

            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = loss(discriminator(real_data), valid)
            fake_loss = loss(discriminator(gen_data.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            if epoch % 20 == 0 and i == 1:
                if cuda:
                    gen_data = gen_data.cpu()
                plt.plot(range(opt.data_length), gen_data[1].detach().numpy())
                plt.savefig("images\\" + str(epoch) + '.jpg')
                plt.close()
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(data_loader), d_loss.item(), g_loss.item())
                )


def gen_images():
    generator = GeneratorFor2D()

    generator.load_state_dict(torch.load('GANParameters\\generator.pt'))

    if cuda:
        generator.cuda()

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    gen_num = 10
    z = Variable(Tensor(np.random.normal(0, 1, (gen_num, opt.latent_dim))))
    gen_data = generator(z)
    if cuda:
        gen_data = gen_data.cpu()
    for i in range(gen_data.shape[0]):
        plt.contourf(gen_data[i][0].detach().numpy())
        plt.axis('off')
        plt.savefig("GenImages\\" + str(i) + '.jpg')
        plt.close()


def test():
    datasets = ReWrite.load_data_in_seq(source_files)
    datasets = ReWrite.MyDataSet(datasets)
    data_loader = DataLoader(
        datasets,
        batch_size=16,
        shuffle=True,
    )
    for i, (imag, labels) in enumerate(data_loader):
        plt.contourf(imag[0])
        plt.show()

    print('done')


def ex_cgan():
    datasets = ReWrite.load_data_in_seq(source_files)
    datasets = ReWrite.MyDataSet(datasets)
    data_loader = DataLoader(
        datasets,
        batch_size=opt.batch_size,
        shuffle=True,
    )
    generator = G_D_Module.GeneratorCGAN(opt.latent_dim, opt.n_classes, img_shape)  # latent_dim should be 200
    discriminator = G_D_Module.DiscriminatorCGAN(opt.n_classes, img_shape)

    TrainFunction.train_cgan(generator, discriminator, data_loader, opt.n_epochs, opt.lr, opt.b1, opt.b2,
                             opt.latent_dim, opt.n_classes, cuda,
                             fist_train=False)


def ex_cdcgan():
    datasets = ReWrite.load_data_in_seq(source_files)
    datasets = ReWrite.MyDataSet(datasets)
    data_loader = DataLoader(
        datasets,
        batch_size=256,
        shuffle=True,
    )
    latent_dim = 20
    generator = G_D_Module.GeneratorCDCGAN(latent_dim, opt.n_classes, img_shape)  # latent_dim should be 20
    discriminator = G_D_Module.DiscriminatorCDCGAN(opt.n_classes, img_shape, latent_dim)

    TrainFunction.train_cdcgan(generator, discriminator, data_loader, opt.n_epochs, opt.lr, opt.b1, opt.b2,
                               opt.latent_dim, opt.n_classes, cuda,
                               fist_train=False)


def ex_wcgan():
    data_sets = ReWrite.load_data_in_seq(source_files)
    data_sets = ReWrite.MyDataSet(data_sets)
    data_loader = DataLoader(
        data_sets,
        batch_size=512,
        shuffle=True,
    )
    latent_dim = 20
    generator = G_D_Module.GeneratorWCGAN(latent_dim, opt.n_classes, img_shape)  # latent_dim should be 20
    discriminator = G_D_Module.DiscriminatorWCGAN(opt.n_classes, img_shape)

    TrainFunction.train_wcgan(generator, discriminator, data_loader, opt.n_epochs, opt.lr, opt.b1, opt.b2,
                              opt.latent_dim, opt.n_classes, cuda,
                              fist_train=False)


def ex_dcwcgan():
    data_sets = ReWrite.load_data_in_seq(source_files)
    data_sets = ReWrite.MyDataSet(data_sets)
    data_loader = DataLoader(
        data_sets,
        batch_size=256,
        shuffle=True,
    )
    generator = G_D_Module.GeneratorDCWCGAN(opt.latent_dim, opt.n_classes, img_shape)  # latent_dim should be 20
    discriminator = G_D_Module.DiscriminatorDCWCGAN(opt.n_classes, img_shape)

    TrainFunction.train_dcwcgan(generator, discriminator, data_loader, opt.n_epochs, opt.lr, opt.b1, opt.b2,
                                opt.latent_dim, opt.n_classes, cuda,
                                fist_train=False)


def ex_ponodcwcgan():
    data_sets = ReWrite.load_data_in_seq(source_files)
    data_sets = ReWrite.MyDataSet(data_sets)
    data_loader = DataLoader(
        data_sets,
        batch_size=256,
        shuffle=True,
    )
    latent_dim = 100
    generator = G_D_Module.GeneratorPONODCWCGAN(latent_dim, opt.n_classes, img_shape)  # latent_dim should be 20
    discriminator = G_D_Module.DiscriminatorPONODCWCGAN(opt.n_classes, img_shape)

    TrainFunction.train_ponodcwcgan(generator, discriminator, data_loader, opt.n_epochs, opt.lr, opt.b1, opt.b2,
                                    latent_dim, opt.n_classes, cuda,
                                    fist_train=False)


def ex_self_noise_gan():
    data_sets = ReWrite.load_data_in_seq(source_files)
    data_sets = ReWrite.MyDataSet(data_sets)
    data_loader = DataLoader(
        data_sets,
        batch_size=256,
        shuffle=True,
    )
    generator = G_D_Module.GeneratorSelfNoise(img_shape)
    discriminator = G_D_Module.DiscriminatorSelfNoise(img_shape)

    TrainFunction.train_self_noise_gan(generator, discriminator, data_loader, opt.n_epochs, opt.lr, opt.b1, opt.b2,
                                       cuda, first_train=False)


def ex_info_gan():
    data_sets = ReWrite.load_data_in_seq(source_files)
    data_sets = ReWrite.MyDataSet(data_sets)
    data_loader = DataLoader(
        data_sets,
        batch_size=256,
        shuffle=True,
    )
    generator = G_D_Module.GeneratorInfo(latent_dim=50, n_classes=5, code_dim=2, img_shape=img_shape)
    discriminator = G_D_Module.DiscriminatorInfo(n_classes=5, code_dim=2, img_shape=img_shape)

    TrainFunction.train_info_gan(generator, discriminator, data_loader, opt.n_epochs, opt.lr, opt.b1, opt.b2,
                                 latent_dim=50, n_classes=5, code_dim=2,
                                 cuda=cuda, first_train=False)


def ex_conv1d_gan():
    data_sets = ReWrite.load_data_in_seq_1d('data')
    data_sets = ReWrite.MyDataSet1D(data_sets)
    data_loader = DataLoader(
        data_sets,
        batch_size=512,
        shuffle=True,
    )
    latent_dim = 50
    generator = G_D_Module.GeneratorConv1D(latent_dim, opt.n_classes)
    discriminator = G_D_Module.DiscriminatorConv1D(opt.n_classes)

    TrainFunction.train_conv1d_gan(generator, discriminator, data_loader, opt.n_epochs, opt.lr, opt.b1, opt.b2,
                                   latent_dim, opt.n_classes, cuda,
                                   first_train=False)


def ex_linear1d_gan():
    data_sets = ReWrite.load_data_in_seq_1d('data')
    data_sets = ReWrite.MyDataSet1D(data_sets)
    data_loader = DataLoader(
        data_sets,
        batch_size=512,
        shuffle=True,
    )
    latent_dim = 50
    generator = G_D_Module.GeneratorLinear1D(latent_dim, opt.n_classes)
    discriminator = G_D_Module.DiscriminatorLinear1D(opt.n_classes)

    TrainFunction.train_linear_1d_gan(generator, discriminator, data_loader, opt.n_epochs, opt.lr, opt.b1, opt.b2,
                                      latent_dim, opt.n_classes, cuda,
                                      first_train=False)


def ex_selfnoise_1d_gan():
    data_sets = ReWrite.load_data_in_seq_1d('data')
    data_sets = ReWrite.MyDataSet1D(data_sets)
    data_loader = DataLoader(
        data_sets,
        batch_size=128,
        shuffle=True,
    )
    generator = G_D_Module.GeneratorSelfNoise1D()
    discriminator = G_D_Module.DiscriminatorSelfNoise1D()

    TrainFunction.train_selfnoise_1d_gan(generator, discriminator, data_loader, opt.n_epochs, opt.lr, opt.b1, opt.b2,
                                         -1, opt.n_classes, cuda,
                                         first_train=False)


if __name__ == '__main__':
    ex_selfnoise_1d_gan()
