import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.image as pltImage
from dataToCoe import data_read
import random
from MyModule import G_D_Module
import torch
import os

# from torch.autograd import Variable


plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def read_csv_data():
    origin_list = os.listdir('data')
    data_list = []  # csv data list
    for name in origin_list:
        if name[-3:] == 'csv':
            data_list.append(name)

    data_length = 1024
    data_np_list = []
    for i in range(len(data_list)):
        source_data = np.loadtxt('data/' + data_list[i])
        length = len(source_data)
        data_count = length // data_length
        data = source_data[:data_count * data_length]
        data = np.reshape(data, (data_count, data_length))
        data_np_list.append(data)
    return data_np_list


def show_cgan_data():
    latent_dim = 20
    data_list = os.listdir('coedatas')
    data = []
    for path in data_list:
        data.append(data_read('coedatas/' + path))

    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    generator = G_D_Module.GeneratorCGAN(latent_dim, 5, (1, 32, 32))
    generator.load_state_dict(torch.load('GANParameters/CGAN/generator.pt'))

    noise = FloatTensor(np.random.normal(0, 1, (len(data) ** 2, latent_dim)))
    single_list = list(range(len(data)))
    label = LongTensor(single_list * len(data))
    gen_imags = generator(noise, label)

    # real
    imgs = np.empty([len(data) ** 2, 1, 32, 32], dtype=float)
    for i in range(len(data)):
        for j in range(len(data)):
            index = random.randint(0, len(data[j]) - 1)
            imgs[i * len(data) + j][0] = data[j][index]
    for i in range(imgs.shape[0]):
        plt.subplot(len(data_list), len(data_list), i + 1)
        plt.axis('off')
        plt.contourf(imgs[i][0])
    plt.savefig('caches/real.jpg', bbox_inches='tight')
    plt.close()

    for i in range(gen_imags.shape[0]):
        plt.subplot(len(data), len(data), i + 1)
        plt.axis('off')
        plt.contourf(gen_imags[i][0].detach().numpy())
    plt.savefig('caches/gen.jpg', bbox_inches='tight')
    plt.close()


def show_cdcgan_data():
    latent_dim = 20
    data_list = os.listdir('coedatas')
    data = []
    for path in data_list:
        data.append(data_read('coedatas/' + path))

    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    generator = G_D_Module.GeneratorCDCGAN(latent_dim, 5, (1, 32, 32))
    generator.load_state_dict(torch.load('GANParameters/CDCGAN/generator.pt'))

    noise = FloatTensor(np.random.normal(0, 1, (len(data) ** 2, latent_dim)))
    single_list = list(range(len(data)))
    label = LongTensor(single_list * len(data))
    gen_imags = generator(noise, label)

    # real
    # imgs = np.empty([len(data) ** 2, 1, 32, 32], dtype=float)
    # for i in range(len(data)):
    #     for j in range(len(data)):
    #         index = random.randint(0, len(data[j]) - 1)
    #         imgs[i * len(data) + j][0] = data[j][index]
    # for i in range(imgs.shape[0]):
    #     plt.subplot(len(data_list), len(data_list), i + 1)
    #     plt.axis('off')
    #     plt.contourf(imgs[i][0])
    # plt.savefig('caches/real.jpg', bbox_inches='tight')
    # plt.close()

    for i in range(gen_imags.shape[0]):
        plt.subplot(len(data), len(data), i + 1)
        plt.axis('off')
        plt.contourf(gen_imags[i][0].detach().numpy())
    plt.savefig('caches/gen.jpg', bbox_inches='tight')
    plt.close()


def show_wcgan_data():
    im_index = range(9)
    latent_dim = 20
    data_list = os.listdir('coedatas')
    data = []
    for path in data_list:
        data.append(data_read('coedatas/' + path))

    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    generator = G_D_Module.GeneratorWCGAN(latent_dim, 5, (1, 32, 32))
    generator.load_state_dict(torch.load('GANParameters/WCGAN/generator.pt'))

    noise = FloatTensor(np.random.normal(0, 1, (len(data) ** 2, latent_dim)))
    single_list = list(range(len(data)))
    label = LongTensor(single_list * len(data))
    gen_imags = generator(noise, label)
    gen_imags = gen_imags.cpu()

    for i in range(gen_imags.size(0)):
        plt.subplot(len(data), len(data), i + 1)
        plt.axis('off')
        plt.contourf(gen_imags[i][0].detach().numpy())
    plt.savefig('caches/gen.jpg', bbox_inches='tight')
    # plt.show()
    plt.close()

    for i in range(len(data)):
        for j in range(len(data)):
            index = random.randint(0, data[j].shape[0] - 1)
            plt.subplot(len(data), len(data), i * len(data) + j + 1)
            plt.axis('off')
            plt.contourf(data[j][index])
    plt.savefig('caches/real.jpg', bbox_inches='tight')
    # plt.show()
    plt.close()


def show_dcwcgan_data():
    im_index = range(9)
    latent_dim = 20
    data_list = os.listdir('coedatas')
    data = []
    for path in data_list:
        data.append(data_read('coedatas/' + path))

    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    generator = G_D_Module.GeneratorDCWCGAN(latent_dim, 5, (1, 32, 32))
    generator.load_state_dict(torch.load('GANParameters/DCWCGAN/generator.pt'))

    noise = FloatTensor(np.random.normal(0, 1, (len(data) ** 2, latent_dim)))
    single_list = list(range(len(data)))
    label = LongTensor(single_list * len(data))
    gen_imags = generator(noise, label)
    gen_imags = gen_imags.cpu()

    for i in range(gen_imags.size(0)):
        plt.subplot(len(data), len(data), i + 1)
        plt.axis('off')
        plt.contourf(gen_imags[i][0].detach().numpy())
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('caches/gen.jpg', bbox_inches='tight', pad_inches=0)
    # plt.show()
    plt.close()

    for i in range(len(data)):
        for j in range(len(data)):
            index = random.randint(0, data[j].shape[0] - 1)
            plt.subplot(len(data), len(data), i * len(data) + j + 1)
            plt.axis('off')
            plt.contourf(data[j][index])
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('caches/real.jpg', bbox_inches='tight', pad_inches=0)
    # plt.show()
    plt.close()


def show_ponodcwcgan_data():
    latent_dim = 100
    data_list = os.listdir('coedatas')
    data = []
    for path in data_list:
        data.append(data_read('coedatas/' + path))

    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    generator = G_D_Module.GeneratorPONODCWCGAN(latent_dim, 5, (1, 32, 32))
    generator.load_state_dict(torch.load('GANParameters/PONODCWCGAN/generator.pt'))

    noise = FloatTensor(np.random.normal(0, 1, (len(data) ** 2, latent_dim)))
    single_list = list(range(len(data)))
    label = LongTensor(single_list * len(data))
    gen_imags = generator(noise, label)
    gen_imags = gen_imags.cpu()

    for i in range(gen_imags.size(0)):
        plt.subplot(len(data), len(data), i + 1)
        plt.axis('off')
        plt.contourf(gen_imags[i][0].detach().numpy())
    plt.savefig('caches/gen.jpg')
    # plt.show()
    plt.close()

    for i in range(len(data)):
        for j in range(len(data)):
            index = random.randint(0, data[j].shape[0] - 1)
            plt.subplot(len(data), len(data), i * len(data) + j + 1)
            plt.axis('off')
            plt.contourf(data[j][index])
    plt.savefig('caches/real.jpg')
    # plt.show()
    plt.close()


def show_self_noise_data():
    data_list = os.listdir('coedatas')
    data = []
    for path in data_list:
        data.append(data_read('coedatas/' + path))

    FloatTensor = torch.cuda.FloatTensor
    generator = G_D_Module.GeneratorSelfNoise((1, 32, 32))
    generator.load_state_dict(torch.load('GANParameters/SELFNOISEGAN/generator.pt'))
    generator.cuda()

    imgs = np.empty([len(data) ** 2, 1, 32, 32], dtype=float)

    for i in range(len(data)):
        for j in range(len(data)):
            index = random.randint(0, len(data[j]) - 1)
            imgs[i * len(data) + j][0] = data[j][index]

    imgs_torch = FloatTensor(imgs)
    gen_imags = generator(imgs_torch)

    for i in range(gen_imags.size(0)):
        plt.subplot(len(data), len(data), i + 1)
        plt.axis('off')
        plt.contourf(gen_imags[i][0].detach().cpu().numpy())
    plt.savefig('caches/gen.jpg', bbox_inches='tight', pad_inches=0)
    # plt.show()
    plt.close()

    # random
    imgs = np.empty([len(data) ** 2, 1, 32, 32], dtype=float)

    for i in range(len(data)):
        for j in range(len(data)):
            index = random.randint(0, len(data[j]) - 1)
            imgs[i * len(data) + j][0] = data[j][index]

    # r
    for i in range(len(imgs)):
        plt.subplot(len(data), len(data), i + 1)
        plt.axis('off')
        plt.contourf(imgs[i][0])
    plt.savefig('caches/real.jpg', bbox_inches='tight', pad_inches=0.0)

    # plt.show()
    plt.close()


def show_perfprmance_data():
    data_list_real = os.listdir('Performance/real')
    data_list_gen = os.listdir('Performance/gen')
    data_real = []
    for path in data_list_real:
        data_real.append(data_read('Performance/real/' + path))

    data_gen = []
    for path in data_list_gen:
        data_gen.append(data_read('Performance/gen/' + path))
    # for i in data_gen[0]:
    #     plt.contourf(i[0])
    #     plt.show()
    #     plt.close()

    imgs_gen = np.empty([len(data_gen) ** 2, 1, 32, 32], dtype=float)

    for i in range(len(data_gen)):
        for j in range(len(data_gen)):
            index = random.randint(0, len(data_gen[j]) - 1)
            imgs_gen[i * len(data_gen) + j][0] = data_gen[j][index]

    imgs_real = np.empty([len(data_real) ** 2, 1, 32, 32], dtype=float)

    for i in range(len(data_real)):
        for j in range(len(data_real)):
            index = random.randint(0, len(data_real[j]) - 1)
            imgs_real[i * len(data_real) + j][0] = data_real[j][index]

    for i in range(imgs_gen.shape[0]):
        plt.subplot(len(data_gen), len(data_gen), i + 1)
        plt.axis('off')
        plt.contourf(imgs_gen[i][0])
    plt.savefig('caches/gen.jpg', bbox_inches='tight', pad_inches=0.0)
    # plt.show()
    plt.close()

    for i in range(imgs_real.shape[0]):
        plt.subplot(len(data_real), len(data_real), i + 1)
        plt.axis('off')
        plt.contourf(imgs_real[i][0])
    plt.savefig('caches/real.jpg', bbox_inches='tight', pad_inches=0.0)
    # plt.show()
    plt.close()


def show_1d_data():
    real_data = read_csv_data()
    n_class = len(real_data)
    weight = 2
    index = np.random.randint(450, size=(5, 2))
    for i in range(n_class):
        for j in range(weight):
            plt.subplot(n_class, weight, i * weight + j + 1)
            plt.xticks([])
            # index = random.randint(0, real_data[i].shape[0] - 1)
            plt.plot(real_data[i][index[i][j]])
    plt.savefig('caches/real_1d_data.jpg', bbox_inches='tight')
    plt.close()

    cuda = torch.cuda.is_available()
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
    latent_dim = 50
    generator = G_D_Module.GeneratorConv1D(latent_dim, n_class)
    generator.load_state_dict(torch.load('GANParameters/CONV1DGAN/generator.pt'))
    if cuda:
        generator.cuda()
    noise = FloatTensor(np.random.normal(0, 1, (n_class * weight, 1, latent_dim)))
    labels = LongTensor(list(range(0, n_class)) * weight)
    gen_data = generator(noise, labels)
    gen_data = gen_data.cpu().detach().numpy()
    for i in range(gen_data.shape[0]):
        plt.subplot(n_class, weight, i + 1)
        plt.xticks([])
        # plt.xlabel(str(i // weight))
        plt.plot(gen_data[i % n_class][0])
    plt.savefig('caches/gen_conv_1d_data.jpg', bbox_inches='tight')
    plt.close()

    generator = G_D_Module.GeneratorSelfNoise1D()
    if cuda:
        generator.cuda()
    inputs = np.empty((n_class * weight, 1, 1024), dtype=float)
    for i in range(inputs.shape[0]):
        # index = random.randint(0, real_data[i // weight].shape[0] - 1)
        inputs[i][0] = real_data[i // weight][index[i // 2][i % 2]]
    inputs = FloatTensor(inputs)
    gen_data = generator(inputs)
    gen_data = gen_data.cpu().detach().numpy()
    for i in range(gen_data.shape[0]):
        plt.subplot(n_class, weight, i + 1)
        plt.xticks([])
        # plt.xlabel(str(i // weight))
        plt.plot(gen_data[i][0])
    plt.savefig('caches/gen_selfnoise_1d_data.jpg', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    show_dcwcgan_data()
