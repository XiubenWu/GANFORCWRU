import numpy as np
import torch
from torch.nn import init

import torch.nn as nn


class GeneratorCGAN(nn.Module):
    def __init__(self, latent_dim, n_classes, img_shape):
        '''
        :param latent_dim: length of noise  opt.latent_dim
        :param n_classes: num of class of data (labels)  opt.n_classes
        :param img_shape: turtle (channels,img size,img size)
        '''
        super(GeneratorCGAN, self).__init__()

        self.img_shape = img_shape

        self.label_emb = nn.Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
        )

    def forward(self, noise, labels):
        '''
        :param noise:
        :param labels:
        :return: (btach size,channels,image size,image size)
        '''
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img


class DiscriminatorCGAN(nn.Module):
    def __init__(self, n_classes, img_shape):
        '''
        :param n_classes: num of class of data (labels)  opt.n_classes
        :param img_shape: turtle (channels,img size,img size)
        '''
        super(DiscriminatorCGAN, self).__init__()

        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_input = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_input)
        return validity


class GeneratorCDCGAN(nn.Module):
    def __init__(self, latent_dim, n_classes, img_shape):
        super(GeneratorCDCGAN, self).__init__()

        self.img_shape = img_shape
        self.label_emb = nn.Embedding(n_classes, 100)
        self.l1 = nn.Linear(latent_dim + 100, img_shape[-1] ** 2)

        self.channel1 = 32
        self.channel2 = 64
        self.channel3 = 64

        self.conv = nn.Sequential(
            nn.BatchNorm2d(self.img_shape[0]),

            nn.Conv2d(self.img_shape[0], self.channel1, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.channel1, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.channel1, self.channel2, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.channel2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.channel2, self.channel3, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.channel3, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel3, self.img_shape[0], 3, stride=1, padding=1),
        )

    def forward(self, noise, labels):
        em_labels = self.label_emb(labels)
        inputs = torch.cat((em_labels, noise), -1)
        inputs = self.l1(inputs)
        out = inputs.view(inputs.size(0), *self.img_shape)
        img = self.conv(out)
        return img


class DiscriminatorCDCGAN(nn.Module):
    def __init__(self, n_classes, img_shape, latent_dim):
        super(DiscriminatorCDCGAN, self).__init__()

        self.img_shape = img_shape
        self.em_label = nn.Embedding(n_classes, latent_dim)
        self.l1 = nn.Linear(latent_dim + img_shape[1] * img_shape[2], 256)
        self.l2 = nn.Linear(256, img_shape[1] * img_shape[2])

        self.channel1 = 32
        self.channel2 = 64
        self.channel3 = 16
        self.conv = nn.Sequential(
            nn.Conv2d(img_shape[0], self.channel1, 3, 1, 0),  # 32 to 30
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(2),  # 30 to 15

            nn.Conv2d(self.channel1, self.channel2, 4, 1, 0),  # 15 to 12
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(2),  # 12 to 6

            nn.Conv2d(self.channel2, self.channel3, 3, 1, 1),

        )
        self.l3 = nn.Sequential(
            nn.Linear(6 * 6 * self.channel3, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs, labels):
        inputs = torch.cat((self.em_label(labels), inputs.view(inputs.size(0), -1)), -1)
        inputs = self.l1(inputs)
        inputs = self.l2(inputs)
        inputs = inputs.view(inputs.size(0), *self.img_shape)
        inputs = self.conv(inputs)
        out = inputs.view(inputs.shape[0], -1)
        valid = self.l3(out)

        return valid


class GeneratorWCGAN(nn.Module):
    def __init__(self, latent_dim, n_classes, img_shape):
        '''
        :param latent_dim: length of noise  opt.latent_dim
        :param n_classes: num of class of data (labels)  opt.n_classes
        :param img_shape: turtle (channels,img size,img size)
        note:feature_dim must be changed in both of G and D
        '''
        super(GeneratorWCGAN, self).__init__()

        feature_dim = 20
        channels1 = 256
        channels2 = 512
        channels3 = 1024
        channels4 = 2048
        self.img_shape = img_shape
        self.label_emb = nn.Embedding(n_classes, feature_dim)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + feature_dim, channels1, normalize=False),
            *block(channels1, channels2),
            *block(channels2, channels3),
            *block(channels3, channels4),
            nn.Linear(channels4, int(np.prod(img_shape))),
        )

    def forward(self, noise, labels):
        '''
        :param noise:
        :param labels:
        :return: (btach size,channels,image size,image size)
        '''
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img


class DiscriminatorWCGAN(nn.Module):
    def __init__(self, n_classes, img_shape):
        '''
        :param n_classes: num of class of data (labels)  opt.n_classes
        :param img_shape: turtle (channels,img size,img size)
        note:feature_dim must be changed in both of G and D
        '''
        super(DiscriminatorWCGAN, self).__init__()

        feature_dim = 20
        channels1 = 256
        channels2 = 1024
        channels3 = 2048
        channels4 = 1024
        self.label_embedding = nn.Embedding(n_classes, feature_dim)

        self.model = nn.Sequential(
            nn.Linear(feature_dim + int(np.prod(img_shape)), channels1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(channels1, channels2),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(channels2, channels3),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(channels3, 1)
        )

    def forward(self, img, labels):
        d_input = torch.cat((self.label_embedding(labels), img.view(img.size(0), -1)), -1)
        validity = self.model(d_input)
        return validity


class GeneratorDCWCGAN(nn.Module):
    def __init__(self, latent_dim, n_classes, img_shape):
        '''
        :param latent_dim: length of noise  opt.latent_dim
        :param n_classes: num of class of data (labels)  opt.n_classes
        :param img_shape: turtle (channels,img size,img size)
        note:feature_dim must be changed in both of G and D
        '''
        super(GeneratorDCWCGAN, self).__init__()

        feature_dim = 20

        channel1 = 64
        channel2 = 128
        channel3 = 256

        self.schannels = 8
        self.img_shape = img_shape

        self.label_emb = nn.Embedding(n_classes, feature_dim)

        self.init_size = img_shape[-1] // 4  # upsample * 2
        self.l1 = nn.Sequential(nn.Linear(latent_dim + feature_dim, self.schannels * self.init_size ** 2))

        self.model = nn.Sequential(
            nn.BatchNorm2d(self.schannels),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(self.schannels, channel1, 3, stride=1, padding=1),
            nn.BatchNorm2d(channel1, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(channel1, channel2, 3, stride=1, padding=1),
            nn.BatchNorm2d(channel2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(channel2, channel3, 3, stride=1, padding=1),
            nn.BatchNorm2d(channel3, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel3, img_shape[0], 3, stride=1, padding=1),
        )

    def forward(self, noise, labels):
        '''
        :param noise:
        :param labels:
        :return: (btach size,channels,image size,image size)
        '''
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        gen_input = self.l1(gen_input)
        gen_input = gen_input.view(gen_input.size(0), self.schannels, self.init_size, self.init_size)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img


class DiscriminatorDCWCGAN(nn.Module):
    def __init__(self, n_classes, img_shape):
        '''
        :param n_classes: num of class of data (labels)  opt.n_classes
        :param img_shape: turtle (channels,img size,img size)
        note:feature_dim must be changed in both of G and D
        '''
        super(DiscriminatorDCWCGAN, self).__init__()

        feature_dim = 20
        channel1 = 64
        channel2 = 128
        channel3 = 256

        self.img_shape = img_shape
        self.em_label = nn.Embedding(n_classes, feature_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(1, channel1, 3, 1, 0),  # 32* to 30*
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(2),  # 30* to 15*

            nn.Conv2d(channel1, channel2, 4, 1, 0),  # 15* to 12*
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(2),  # 12* to 6*

            nn.Conv2d(channel2, channel3, 3, 1, 1),  # 6* to 6*
        )
        self.l1 = nn.Sequential(
            nn.Linear(6 * 6 * channel3 + feature_dim, 100),
            nn.Linear(100, 1)

        )

    def forward(self, inputs, labels):
        emb_input = self.em_label(labels)
        img_input = self.conv(inputs)
        img_input = img_input.view(img_input.size(0), -1)  # [img_input.size(0),6*6*channel2]
        inputs = torch.cat((emb_input, img_input), -1)
        valid = self.l1(inputs)

        return valid


'''
This is a place for PONO GAN code up...........................................................................
...............................................................................................................
...............................................................................................................
...............................................................................................................
...............................................................................................................
...............................................................................................................
...............................................................................................................
'''


class GeneratorPONODCWCGAN(nn.Module):
    def __init__(self, latent_dim, n_classes, img_shape):
        '''
        :param latent_dim: length of noise  opt.latent_dim
        :param n_classes: num of class of data (labels)  opt.n_classes
        :param img_shape: turtle (channels,img size,img size)
        note:feature_dim must be changed in both of G and D
        '''
        super(GeneratorPONODCWCGAN, self).__init__()

        feature_dim = 20

        channel0 = 16
        channel1 = 64
        channel2 = 128
        channel3 = 256

        self.schannels = 2
        self.img_shape = img_shape

        self.label_emb = nn.Embedding(n_classes, feature_dim)

        self.init_size = img_shape[-1] // 1  # upsample * 0
        self.l1 = nn.Sequential(nn.Linear(latent_dim + feature_dim, self.schannels * self.init_size ** 2))

        self.conv0 = nn.Sequential(
            nn.BatchNorm2d(self.schannels, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.schannels, channel0, 3, stride=1, padding=1)
        )

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(channel0, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Upsample(scale_factor=2),
            nn.Conv2d(channel0, channel1, 3, stride=1, padding=1)
        )

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(channel1, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Upsample(scale_factor=2),
            nn.Conv2d(channel1, channel2, 3, stride=1, padding=1)
        )

        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(channel2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel2, channel3, 3, stride=1, padding=1)
        )

        self.conv4 = nn.Sequential(
            nn.BatchNorm2d(channel3, 0.8),
            nn.Conv2d(channel3, img_shape[0], 3, stride=1, padding=1)
        )

        # self.model = nn.Sequential(
        #     nn.BatchNorm2d(self.schannels),  ###########
        #     nn.Upsample(scale_factor=2),
        #
        #     nn.Conv2d(self.schannels, channel1, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(channel1, 0.8),  #############
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Upsample(scale_factor=2),
        #
        #     nn.Conv2d(channel1, channel2, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(channel2, 0.8),  ###########
        #     nn.LeakyReLU(0.2, inplace=True),
        #
        #     nn.Conv2d(channel2, channel3, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(channel3, 0.8),  #############
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(channel3, img_shape[0], 3, stride=1, padding=1),
        # )

    def PONO(self, x, eps=0.00001):
        mean = x.mean(dim=1, keepdim=True)
        std = (torch.var(x, dim=1, keepdim=True) + eps).sqrt()
        x = (x - mean) / std
        return x, mean, std

    def MS(self, x, beta, gamma):
        '''Decoding
        :param x: inputs
        :param beta: mean
        :param gamma: std
        :return: processed x
        '''
        x.mul_(gamma)
        x.add_(beta)
        return x

    def forward(self, noise, labels):
        '''
        :param noise:
        :param labels:
        :return: (btach size,channels,image size,image size)
        '''
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        gen_input = self.l1(gen_input)
        gen_input = gen_input.view(gen_input.size(0), self.schannels, self.init_size, self.init_size)

        x = self.conv0(gen_input)
        x, mean, std = self.PONO(x)
        x = self.conv1(x)
        x = self.MS(x, mean, std)

        x, mean, std = self.PONO(x)
        x = self.conv2(x)
        x = self.MS(x, mean, std)

        x, mean, std = self.PONO(x)
        x = self.conv3(x)
        x = self.MS(x, mean, std)

        img = self.conv4(x)

        img = img.view(img.size(0), *self.img_shape)
        return img


class DiscriminatorPONODCWCGAN(nn.Module):
    def __init__(self, n_classes, img_shape):
        '''
        :param n_classes: num of class of data (labels)  opt.n_classes
        :param img_shape: turtle (channels,img size,img size)
        note:feature_dim must be changed in both of G and D
        '''
        super(DiscriminatorPONODCWCGAN, self).__init__()

        feature_dim = 20
        channel1 = 64
        channel2 = 128
        channel3 = 256

        self.img_shape = img_shape
        self.em_label = nn.Embedding(n_classes, feature_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(1, channel1, 3, 1, 0),  # 32* to 30*
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(2),  # 30* to 15*

            nn.Conv2d(channel1, channel2, 4, 1, 0),  # 15* to 12*
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(2),  # 12* to 6*

            nn.Conv2d(channel2, channel3, 3, 1, 1),  # 6* to 6*
        )
        self.l1 = nn.Sequential(
            nn.Linear(6 * 6 * channel3 + feature_dim, 100),
            nn.Linear(100, 1)

        )

    def forward(self, inputs, labels):
        emb_input = self.em_label(labels)
        img_input = self.conv(inputs)
        img_input = img_input.view(img_input.size(0), -1)  # [img_input.size(0),6*6*channel2]
        inputs = torch.cat((emb_input, img_input), -1)
        valid = self.l1(inputs)

        return valid

    def PONO(self, x, eps=0.00001):
        mean = x.mean(dim=1, keepdim=True)
        std = (torch.var(x, dim=1, keepdim=True) + eps).sqrt()
        x = (x - mean) / std
        return x, mean, std

    def MS(self, x, beta, gamma):
        '''Decoding
        :param x: inputs
        :param beta: mean
        :param gamma: std
        :return: processed x
        '''
        x.mul_(gamma)
        x.add_(beta)
        return x


'''
This is a place for PONO GAN code down...........................................................................
...............................................................................................................
...............................................................................................................
...............................................................................................................
...............................................................................................................
...............................................................................................................
...............................................................................................................

'''


class GeneratorSelfNoise(nn.Module):
    def __init__(self, img_shape):
        '''
        # :param latent_dim: length of noise  opt.latent_dim
        # :param n_classes: num of class of data (labels)  opt.n_classes
        :param img_shape: turtle (channels,img size,img size)
        note:feature_dim must be changed in both of G and D
        '''
        super(GeneratorSelfNoise, self).__init__()

        channel0 = 4
        channel1 = 16
        channel2 = 32
        channel3 = 64
        channel4 = 128

        self.img_shape = img_shape

        self.extend = nn.Conv2d(1, channel0, 3, 1, 1)

        self.convDown = nn.Sequential(
            nn.Conv2d(channel0 + 1, channel1, 3, 1, 1),
            nn.BatchNorm2d(channel1, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2)
        )
        self.convUp = nn.Sequential(
            nn.Conv2d(channel1, channel2, 3, 1, 1),
            nn.BatchNorm2d(channel2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2)
        )

        self.convNormal = nn.Sequential(
            nn.Conv2d(channel2, channel3, 3, 1, 1),
            nn.BatchNorm2d(channel3, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(channel3, channel4, 3, 1, 1),
            nn.BatchNorm2d(channel4, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(channel4, img_shape[0], 3, 1, 1)
        )

    def forward(self, inputs):
        x = self.extend(inputs)
        x, mean, std = self.PONO(x)
        noise = torch.randn(inputs.shape[0], 1, self.img_shape[1], self.img_shape[2]).cuda()
        x = torch.cat((x, noise), dim=1)
        x = self.convDown(x)
        x = self.convUp(x)
        x = self.MS(x, mean, std)
        imgs = self.convNormal(x)
        return imgs

    def PONO(self, x, eps=0.00001):
        mean = x.mean(dim=1, keepdim=True)
        std = (torch.var(x, dim=1, keepdim=True) + eps).sqrt()
        x = (x - mean) / std
        return x, mean, std

    def MS(self, x, mean, std):
        '''Decoding
        :param x: inputs
        :param mean: mean
        :param std: std
        :return: processed x
        '''
        x.mul_(std)
        x.add_(mean)
        return x


class DiscriminatorSelfNoise(nn.Module):
    def __init__(self, img_shape):
        '''
        :param img_shape: turtle (channels,img size,img size)
        note:feature_dim must be changed in both of G and D
        '''
        super(DiscriminatorSelfNoise, self).__init__()

        channel1 = 64
        channel2 = 128
        channel3 = 256

        self.img_shape = img_shape
        self.conv = nn.Sequential(
            nn.Conv2d(self.img_shape[0], channel1, 3, 1, 0),  # 32* to 30*
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(2),  # 30* to 15*

            nn.Conv2d(channel1, channel2, 4, 1, 0),  # 15* to 12*
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(2),  # 12* to 6*

            nn.Conv2d(channel2, channel3, 3, 1, 1),  # 6* to 6*
        )
        self.l1 = nn.Sequential(
            nn.Linear(6 * 6 * channel3, 100),
            nn.Linear(100, 1)

        )

    def forward(self, inputs):
        img_input = self.conv(inputs)
        img_input = img_input.view(img_input.size(0), -1)
        valid = self.l1(img_input)
        return valid


class GeneratorInfo(nn.Module):
    def __init__(self, latent_dim, n_classes, code_dim, img_shape):
        super(GeneratorInfo, self).__init__()

        input_dim = latent_dim + n_classes + code_dim

        channels1 = 128
        channels2 = 64
        channels3 = 32

        self.channels1 = channels1

        self.init_size = img_shape[2] // 4  # 2 UpSampling
        self.l1 = nn.Sequential(nn.Linear(input_dim, channels1 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(channels1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(channels1, channels2, 3, stride=1, padding=1),
            nn.BatchNorm2d(channels2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(channels2, channels3, 3, stride=1, padding=1),
            nn.BatchNorm2d(channels3, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels3, img_shape[0], 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels, code):
        gen_input = torch.cat((noise, labels, code), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], self.channels1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class DiscriminatorInfo(nn.Module):
    def __init__(self, n_classes, code_dim, img_shape):
        super(DiscriminatorInfo, self).__init__()

        channels1 = 16
        channels2 = 32
        channels3 = 64
        channels4 = 128

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 1, 0), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(img_shape[0], channels1, bn=False),  # 32 to 30
            *discriminator_block(channels1, channels2),  # 30 to 28
            nn.MaxPool2d(2),  # 28 to 14
            *discriminator_block(channels2, channels3),  # 14 to 12
            *discriminator_block(channels3, channels4)  # 12 to 10
        )

        self.adv_layer = nn.Sequential(nn.Linear(channels4 * 10 ** 2, 1))
        self.aux_layer = nn.Sequential(nn.Linear(channels4 * 10 ** 2, n_classes), nn.Softmax(dim=1))
        self.latent_layer = nn.Sequential(nn.Linear(channels4 * 10 ** 2, code_dim))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        latent_code = self.latent_layer(out)

        return validity, label, latent_code


class GeneratorConv1D(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super(GeneratorConv1D, self).__init__()

        feature_dim = 2 * n_classes
        channels1 = 64
        channels2 = 128
        channels3 = 64

        self.embeding_layer = nn.Embedding(n_classes, feature_dim)
        self.l1 = nn.Linear(latent_dim + feature_dim, 1024 + 4 * 4)
        self.conv1d_layers = nn.Sequential(
            nn.Conv1d(1, channels1, 5, 1),
            nn.BatchNorm1d(channels1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(channels1, channels2, 5, 1),
            nn.BatchNorm1d(channels2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(channels2, channels3, 5, 1),
            nn.BatchNorm1d(channels3),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(channels3, 1, 5, 1)
        )

    def forward(self, noise, labels):
        labels_embeding = self.embeding_layer(labels)
        labels_embeding = labels_embeding.view(labels_embeding.shape[0], 1, -1)
        inputs = torch.cat((noise, labels_embeding), -1)
        inputs = self.l1(inputs)
        datas = self.conv1d_layers(inputs)
        return datas


class DiscriminatorConv1D(nn.Module):
    def __init__(self, n_classes):
        super(DiscriminatorConv1D, self).__init__()

        feature_dim = 2 * n_classes
        channels1 = 32
        channels2 = 64
        channels3 = 32

        self.embeding_layer = nn.Embedding(n_classes, feature_dim)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, channels1, 11, 1),
            nn.BatchNorm1d(channels1),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(channels1, channels2, 11, 1),
            nn.BatchNorm1d(channels2),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(channels2, channels3, 11, 1),
            nn.BatchNorm1d(channels3),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(channels3, 1, 11, 1)
        )
        self.l1 = nn.Sequential(
            nn.Linear(1024 - 10 * 4 + feature_dim, 100),
            nn.Linear(100, 1)
        )

    def forward(self, datas, labels):
        labels_embeding = self.embeding_layer(labels)
        datas = self.conv_layers(datas)
        datas = datas.view(datas.shape[0], -1)
        datas = torch.cat((datas, labels_embeding), -1)
        valids = self.l1(datas)
        return valids


class GeneratorLinear1D(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super(GeneratorLinear1D, self).__init__()

        nodes1 = 512
        nodes2 = 256
        nodes3 = 512
        nodes4 = 1024

        self.em_layer = nn.Embedding(n_classes, n_classes * 2)
        self.layer = nn.Sequential(
            nn.Linear(latent_dim + n_classes * 2, nodes1),
            nn.BatchNorm1d(nodes1, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nodes1, nodes2),
            nn.BatchNorm1d(nodes2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nodes2, nodes3),
            nn.BatchNorm1d(nodes3, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nodes3, nodes4)
        )

    def forward(self, noise, labels):
        inputs = torch.cat((noise, self.em_layer(labels)), dim=-1)
        return self.layer(inputs)


class DiscriminatorLinear1D(nn.Module):
    def __init__(self, n_class):
        super(DiscriminatorLinear1D, self).__init__()

        nodes1 = 512
        nodes2 = 256
        nodes3 = 512

        self.em_layers = nn.Embedding(n_class, 2 * n_class)
        self.layers = nn.Sequential(
            nn.Linear(n_class * 2 + 1024, nodes1),
            nn.BatchNorm1d(nodes1, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nodes1, nodes2),
            nn.BatchNorm1d(nodes2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nodes2, nodes3),
            nn.BatchNorm1d(nodes3, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nodes3, 1)
        )

    def forward(self, data, labels):
        inputs = torch.cat((data, self.em_layers(labels)), -1)
        return self.layers(inputs)


class GeneratorSelfNoise1D(nn.Module):
    def __init__(self):
        super(GeneratorSelfNoise1D, self).__init__()

        channel0 = 8
        channel1 = 32
        channel2 = 64
        channel3 = 32

        self.extent = nn.Conv1d(1, channel0, 1, padding=2)  # 1028

        self.l1 = nn.Linear(100, 1028)

        self.models = nn.Sequential(
            nn.Conv1d(channel0 + 1, channel1, 3, padding=1),  # 1028
            nn.BatchNorm1d(channel1, momentum=0.8),
            nn.ELU(),

            # nn.Conv1d(channel1, channel2, 7, padding=0),  # 1030
            # nn.BatchNorm1d(channel2, 0.8),
            # nn.LeakyReLU(0.2, inplace=True),
        )
        self.l2 = nn.Sequential(
            nn.Conv1d(channel1, channel2, 3, padding=0),  # 1028
            nn.BatchNorm1d(channel2, momentum=0.8),

            nn.Conv1d(channel2, 1, 3, padding=0),  # 1026
        )
        self.init_params()

    def forward(self, inputs):
        x = self.extent(inputs)
        x, mean, std = self.pono(x)
        noise = torch.randn(x.shape[0], 100)
        if torch.cuda.is_available():
            noise = noise.cuda()
        noise = self.l1(noise)
        noise = noise.view(noise.shape[0], 1, -1)
        x = torch.cat((x, noise), dim=1)
        x = self.models(x)
        x = self.ms(x, mean, std)
        x = self.l2(x)
        return x

    def pono(self, x, eps=1e-5):
        mean = x.mean(dim=1, keepdim=True)
        std = (torch.var(x, dim=1, keepdim=True) + eps).sqrt()
        x = (x - mean) / std
        return x, mean, std

    def ms(self, x, mean, std):
        x.mul_(std)
        x.add_(mean)
        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class DiscriminatorSelfNoise1D(nn.Module):
    def __init__(self):
        super(DiscriminatorSelfNoise1D, self).__init__()

        channel1 = 64
        channel2 = 128
        channel3 = 256

        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, channel1, 11),
            nn.BatchNorm1d(channel1, momentum=0.8),
            nn.Dropout(0.2),
            nn.ELU(),
            # nn.Conv1d(channel1, channel2, 11),
            # nn.BatchNorm1d(channel2, momentum=0.8),
            # nn.Dropout(0.2),
            # nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv1d(channel2, channel3, 11),
            # nn.BatchNorm1d(channel3, momentum=0.8),
            # nn.Dropout(0.2),
            # nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(channel1, 1, 11)
        )
        self.l2 = nn.Sequential(
            nn.Linear(1024 - 10 * 2, 100),
            nn.BatchNorm1d(100),
            nn.ELU(),
            nn.Linear(100, 1)
        )

    def forward(self, inputs):
        x = self.conv_layers(inputs)
        x = x.view(x.shape[0], -1)
        x = self.l2(x)
        return x
