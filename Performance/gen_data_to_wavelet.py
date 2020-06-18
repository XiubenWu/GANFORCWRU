import numpy as np
from MyModule import wavelet
import os
from PIL import Image
import random
from matplotlib import pyplot as plt


def read_csv_data():
    origin_list = os.listdir('../data')
    data_list = []  # csv data list
    for name in origin_list:
        if name[-3:] == 'csv':
            data_list.append(name)

    data_length = 1024
    data_np_list = []
    for i in range(len(data_list)):
        source_data = np.loadtxt('../data/' + data_list[i])
        length = len(source_data)
        data_count = length // data_length
        data = source_data[:data_count * data_length]
        data = np.reshape(data, (data_count, data_length))
        data_np_list.append(data)
    return data_np_list


def read_gen_data():
    return np.load('gen_1d/data.npz')['arr_0']


def data_to_wavelet():
    plt.rcParams['image.cmap'] = 'gray'
    data_np = read_gen_data()
    matrix = np.empty([data_np.shape[0] ** 2, 32, 32], dtype=float)
    for i in range(data_np.shape[0]):
        for j in range(data_np.shape[0]):
            index = random.randint(0, data_np[j].shape[0] - 1)
            cos, _ = wavelet.cal_wave(data_np[j][index], 48000, 32 + 1)
            cos = abs(cos)
            im = Image.fromarray(cos)
            im = im.resize((32, 32))
            cos = np.array(im)
            matrix[i * data_np.shape[0] + j] = cos

    for i in range(matrix.shape[0]):
        plt.subplot(data_np.shape[0], data_np.shape[0], i + 1)
        plt.axis('off')
        plt.contourf(matrix[i])
    plt.savefig('../caches/gen_data_to_wavelet.jpg', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    data_to_wavelet()
