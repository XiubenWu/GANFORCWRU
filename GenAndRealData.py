import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataToCoe import data_read
import random
from MyModule import G_D_Module
import os

plt.rcParams['figure.figsize'] = (32, 32)
plt.rcParams['figure.dpi'] = 1
plt.rcParams['image.cmap'] = 'gray'

save_path = 'Performance/'
'''
GenAndRealImgs
Performance
'''

os.makedirs(save_path + 'real', exist_ok=True)
os.makedirs(save_path + 'gen', exist_ok=True)


'''
iner function...........................................................................
'''


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
'''
iner function...........................................................................
'''


def dcwcgan_data(img_epoch_num=50, cuda=True):
    '''
    :param img_epoch_num:
    :param cuda:
    :return: imgs(.jpg) note: change the save path
    '''
    latent_dim = 20  # details in G_D_Module
    n_class = 5  # details in G_D_Module
    img_shape = (1, 32, 32)  # details in G_D_Module
    data_list = os.listdir('coedatas')
    data = []
    for path in data_list:
        data.append(data_read('coedatas/' + path))

    if cuda:
        FloatTensor = torch.cuda.FloatTensor
        LongTensor = torch.cuda.LongTensor

    generator = G_D_Module.GeneratorDCWCGAN(latent_dim, n_class, img_shape)
    generator.load_state_dict(torch.load('GANParameters/DCWCGAN/generator.pt'))

    noise = FloatTensor(np.random.normal(0, 1, (img_epoch_num * n_class, latent_dim)))
    single_list = list(range(n_class))
    label_cpu = single_list * img_epoch_num
    label = LongTensor(label_cpu)
    if cuda:
        label.cuda()
        generator.cuda()
    gen_imags = generator(noise, label)
    gen_imags = gen_imags.cpu()

    for i in range(gen_imags.shape[0]):
        plt.axis('off')
        plt.contourf(gen_imags[i][0].detach().numpy())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.savefig(save_path + 'gen/label' + str(i % 5) + '_' + str(i // 5) + '.jpg')
        plt.close()

    for i in range(img_epoch_num):
        for j in range(len(data)):
            index = random.randint(0, data[j].shape[0] - 1)
            plt.axis('off')
            plt.contourf(data[j][index])
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.savefig(save_path + 'real/label' + str(j) + '_' + str(i) + '.jpg')
            plt.close()


def self_noise_data(img_epoch_num=50, cuda=True):
    '''
        :param img_epoch_num:
        :param cuda:
        :return: imgs(.jpg) note: change the save path
        '''
    img_shape = (1, 32, 32)  # details in G_D_Module
    data_list = os.listdir('coedatas')
    data = []
    for path in data_list:
        data.append(data_read('coedatas/' + path))

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    generator = G_D_Module.GeneratorSelfNoise(img_shape)
    generator.load_state_dict(torch.load('GANParameters/SELFNOISEGAN/generator.pt'))

    if cuda:
        generator.cuda()

    for epoch in range(img_epoch_num):
        imgs = np.empty([len(data), 1, 32, 32], dtype=float)
        for i in range(len(data)):
            index = random.randint(0, len(data[i]) - 20 - 1)
            imgs[i][0] = data[i][index]

        if cuda:
            imgs_torch = FloatTensor(imgs).cuda()
        gen_imags = generator(imgs_torch)
        gen_imags = gen_imags.cpu()

        for i in range(gen_imags.shape[0]):
            plt.axis('off')
            plt.contourf(gen_imags[i][0].detach().numpy())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.savefig(save_path + 'gen/label' + str(i) + '_' + str(epoch) + '.jpg')
            plt.close()

        for i in range(imgs.shape[0]):
            plt.axis('off')
            plt.contourf(imgs[i][0])
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.savefig(save_path + 'real/label' + str(i) + '_' + str(epoch) + '.jpg')
            plt.close()


def dcwcgan_data_perf(img_epoch_num=50, cuda=True):
    '''
    :param img_epoch_num:
    :param cuda:
    :return:
    note: generate the .npy in the Performance
    '''
    latent_dim = 20  # details in G_D_Module
    n_class = 5  # details in G_D_Module
    img_shape = (1, 32, 32)  # details in G_D_Module
    data_list = os.listdir('coedatas')
    data = []
    for path in data_list:
        data.append(data_read('coedatas/' + path))

    if cuda:
        FloatTensor = torch.cuda.FloatTensor
        LongTensor = torch.cuda.LongTensor
    else:
        FloatTensor = torch.FloatTensor
        LongTensor = torch.LongTensor

    generator = G_D_Module.GeneratorDCWCGAN(latent_dim, n_class, img_shape)
    generator.load_state_dict(torch.load('GANParameters/DCWCGAN/generator.pt'))

    noise = FloatTensor(np.random.normal(0, 1, (img_epoch_num * n_class, latent_dim)))
    single_list = list(range(n_class))
    label_cpu = single_list * img_epoch_num
    label = LongTensor(label_cpu)
    if cuda:
        label.cuda()
        generator.cuda()
    gen_imags = generator(noise, label)
    gen_imags = gen_imags.cpu()

    datas = []
    for i in range(len(data_list)):
        datas.append(np.empty((img_epoch_num, *img_shape), dtype=float))

    for i in range(len(label_cpu)):
        datas[label_cpu[i]][i // 5][0] = gen_imags[i][0].detach().numpy()

    for i in range(len(datas)):
        np.savez(save_path + "gen/%d.npz" % i, datas[i])
        print(datas[i].shape)


def self_noise_data_perf(img_epoch_num=50, cuda=True):
    """
    :param img_epoch_num:
    :param cuda:
    :return: .npz in directory of gen
    """
    n_class = 5  # details in G_D_Module
    img_shape = (1, 32, 32)  # details in G_D_Module
    data_list = os.listdir('coedatas')
    data = []
    for path in data_list:
        data.append(data_read('coedatas/' + path))

    if cuda:
        FloatTensor = torch.cuda.FloatTensor
        LongTensor = torch.cuda.LongTensor
    else:
        FloatTensor = torch.FloatTensor
        LongTensor = torch.LongTensor

    generator = G_D_Module.GeneratorSelfNoise(img_shape)
    generator.load_state_dict(torch.load('GANParameters/SELFNOISEGAN/generator.pt'))

    end_list = None
    for i in range(img_epoch_num // 50):
        datas_list = []
        for j in range(n_class):
            mid = np.empty([50, 1, 32, 32], dtype=float)
            for k in range(50):
                mid[k][0] = data[j][random.randint(0, data[j].shape[0] - 1)]
            datas_list.append(mid)
        img_list = []
        for img in datas_list:
            if cuda:
                generator.cuda()
                img = FloatTensor(img)
            gen_imags = generator(img)
            gen_imags = gen_imags.cpu()
            img_list.append(gen_imags)
        if end_list is None:
            end_list = [img_list[mid].detach().numpy() for mid in range(len(img_list))]
        else:
            for count in range(len(end_list)):
                end_list[count] = np.concatenate((end_list[count], img_list[count].detach().numpy()),
                                                 axis=0)

    for i in range(len(end_list)):
        np.savez(save_path + "gen/%d.npz" % i, end_list[i])
        print(end_list[i].shape)

    # for i in range(len(end_list)):
    #     for j in range(end_list[i].shape[0]):
    #         plt.axis('off')
    #         plt.contourf(end_list[i][j][0])
    #         plt.savefig('caches/' + str(i) + '_' + str(j) + '.jpg')
    #         plt.close()


def gan_data_1d_perf(img_epoch_num, cuda=True):
    """This is WCGAN1D to performance
    :param img_epoch_num:
    :param cuda:
    :return: gen_1d/data_wcgan.npz
    """
    latent_dim = 50  # details in G_D_Module
    n_class = 5  # details in G_D_Module

    if cuda:
        FloatTensor = torch.cuda.FloatTensor
        LongTensor = torch.cuda.LongTensor
    else:
        FloatTensor = torch.FloatTensor
        LongTensor = torch.LongTensor

    generator = G_D_Module.GeneratorConv1D(latent_dim, n_class)
    generator.load_state_dict(torch.load('GANParameters/CONV1DGAN/generator.pt'))

    noise = np.random.normal(0, 1, (img_epoch_num * n_class, latent_dim))
    noise = noise[:, np.newaxis, :]
    noise = FloatTensor(noise)
    single_list = list(range(n_class))
    label_cpu = single_list * img_epoch_num
    label = LongTensor(label_cpu)
    if cuda:
        label.cuda()
        generator.cuda()
    gen_datas = generator(noise, label)
    gen_datas = gen_datas.cpu()

    datas = np.empty((n_class, img_epoch_num, 1024), dtype=float)

    for i in range(len(label_cpu)):
        datas[label_cpu[i]][i // 5] = gen_datas[i].detach().numpy()

    np.savez(save_path + "gen_1d/data_wcgan.npz", datas)

    # for i in range(n_class):
    #     for j in range(n_class):
    #         plt.subplot(n_class, n_class, i * n_class + j + 1)
    #         plt.plot(datas[j][random.randint(0, img_epoch_num - 1)])
    # plt.show()
    # plt.close()


def gan_data_selfnoise_1d_perf(data_epoch_num, cuda=True):
    n_class = 5
    if cuda:
        FloatTensor = torch.cuda.FloatTensor
        LongTensor = torch.cuda.LongTensor
    else:
        FloatTensor = torch.FloatTensor
        LongTensor = torch.LongTensor

    generator = G_D_Module.GeneratorSelfNoise1D()
    generator.load_state_dict(torch.load('GANParameters/SELFNOISE1DGAN/generator.pt'))

    if cuda:
        generator.cuda()

    datas = np.empty((n_class, data_epoch_num, 1024), dtype=float)
    data_list = read_csv_data()
    for i in range(data_epoch_num):
        data = np.empty((n_class, 1, 1024), dtype=float)
        for j in range(n_class):
            index = random.randint(0, data_list[j].shape[0] - 1)
            data[j][0] = data_list[j][index]
        data = FloatTensor(data)
        gen_data = generator(data).cpu().detach().numpy()
        for j in range(n_class):
            datas[j][i] = gen_data[j][0]
        print('[%d/%d]' % (i, data_epoch_num))

    np.savez(save_path + "gen_1d/data_selfnoise.npz", datas)


def img_data(img_epoch_num, cuda=True):
    '''
        :param img_epoch_num:
        :param cuda:
        :return: imgs(.jpg) note: change the save path
    '''
    os.makedirs('Performance/imgdatas/train/gen', exist_ok=True)
    os.makedirs('Performance/imgdatas/train/real', exist_ok=True)
    os.makedirs('Performance/imgdatas/test', exist_ok=True)

    img_shape = (1, 32, 32)  # details in G_D_Module
    data_list = os.listdir('coedatas')
    data = []
    for path in data_list:
        data.append(data_read('coedatas/' + path))

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    generator = G_D_Module.GeneratorSelfNoise(img_shape)
    generator.load_state_dict(torch.load('GANParameters/SELFNOISEGAN/generator.pt'))

    if cuda:
        generator.cuda()

    for epoch in range(img_epoch_num):
        imgs = np.empty([len(data), 1, 32, 32], dtype=float)
        for i in range(len(data)):
            index = random.randint(0, len(data[i]) - 20 - 1)
            imgs[i][0] = data[i][index]

        imgs_torch = FloatTensor(imgs)
        gen_imags = generator(imgs_torch)
        gen_imags = gen_imags.cpu()

        for i in range(gen_imags.shape[0]):
            plt.axis('off')
            plt.contourf(gen_imags[i][0].detach().numpy())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.savefig('Performance/imgdatas/train/gen/' + str(i) + '_' + str(epoch) + '.jpg',
                        pad_inches=0.0)
            plt.close()

    # real data 300 for train 170+ for test don't commend if need

    for i in range(len(data)):
        for j in range(data[i].shape[0]):
            if j < 300:
                plt.axis('off')
                plt.contourf(data[i][j])
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.savefig('Performance/imgdatas/train/real/' + str(i) + '_' + str(j) + '.jpg',
                            pad_inches=0.0)
                plt.close()
            else:
                plt.axis('off')
                plt.contourf(data[i][j])
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.savefig('Performance/imgdatas/test/' + str(i) + '_' + str(j) + '.jpg',
                            pad_inches=0.0)
                plt.close()


if __name__ == '__main__':
    gan_data_1d_perf(200)




