import numpy as np
from MyModule import wavelet
import os
import matplotlib.pyplot as plt
from PIL import Image

os.makedirs('coedatas', exist_ok=True)
num = 4
dataPath = r'data/' + str(num) + '.csv'
savePath = 'coedatas/' + str(num) + '.npz'
source_data = np.loadtxt(dataPath)

data_length = 1024


def data_cal(data, sampling_rate, total_scale, wave_name='cgau8'):
    """data: type = np;dim = 2
        from originral data to coefficient of abs
    """
    length = len(data)
    data_count = length // data_length
    data = data[:data_count * data_length]
    data = np.reshape(data, (data_count, data_length))
    coes = None
    for i in range(data.shape[0]):
        coe, _ = wavelet.cal_wave(data[i], sampling_rate=sampling_rate, total_scale=total_scale, wave_name=wave_name)
        coe = coe[np.newaxis, :, :]
        if coes is None:
            coes = coe
        else:
            coes = np.concatenate((coes, coe), axis=0)

    return abs(coes)


def data_write(data, sampling_rate, total_scale, re_size=(32, 32), wave_name='cgau8'):
    '''
    :param data: source data,
    :param sampling_rate: 12kHz or 48kHz
    :param total_scale:default 32
    :param re_size:
    :param wave_name:cgau8
    :return: save a .npz files of recoes
    '''

    coes = data_cal(data, sampling_rate, total_scale, wave_name)
    # plt.contourf(coes[0])
    # plt.show()
    if re_size == None:
        np.savez(savePath, coes)
    else:
        re_coes = None
        for i in range(coes.shape[0]):
            im = Image.fromarray(coes[i])
            im = im.resize(re_size)
            re_coe = np.array(im)
            re_coe = re_coe[np.newaxis, :, :]
            if re_coes is None:
                re_coes = re_coe
            else:
                re_coes = np.concatenate((re_coes, re_coe), axis=0)

        np.savez(savePath, re_coes)


def data_read(filename):
    npzfile = np.load(filename)
    return npzfile['arr_0']


data_write(source_data, 48000, 32, (32, 32))
data = data_read(savePath)

# plt.contourf(data[0])
# plt.show()
