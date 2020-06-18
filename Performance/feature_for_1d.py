import numpy as np
from matplotlib import pyplot as plt
import os
import random
import xlwt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


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


def read_gen_data(name):
    return np.load('gen_1d/' + name + '.npz')['arr_0']


def fft(data, sample_rate=48000):
    f = np.fft.fft(data)
    f = f[0:(f.shape[-1] // 2)]
    f = abs(f)
    f = f / max(f)
    freq = np.fft.fftfreq(data.shape[-1], d=1.0 / sample_rate)
    freq = freq[:(freq.shape[-1] // 2)]

    # plt.plot(data)
    # plt.show()
    # plt.plot(freq, abs(f))
    # plt.show()
    return abs(f), freq


def cal_mean(data_single):
    return np.mean(data_single, axis=1)


def cal_std(data_single):
    return np.std(data_single, axis=1)


def cal_min(data_single):
    return np.min(data_single, axis=1)


def cal_max(data_single):
    return np.max(data_single, axis=1)


def plot_all(mode, flag, *name):
    """ plot feature in caches
        :param mode: 'gen' or 'real'
        :param flag: a data a figure when true; all data a figure when false
        :param name: 'std' 'mean' 'min' 'max'
        :return: plot
    """
    if mode in ('data_selfnoise', 'data_wcgan'):
        data_np_list = read_gen_data(mode)
    elif mode == 'real':
        data_np_list = read_csv_data()
    else:
        raise ValueError('input mode error')

    if 'std' in name:
        plt.title('各类数据标准差分布')
        plt.xlabel('')
        plt.ylabel('标准差')
        if flag:
            for i in range(len(data_np_list)):
                std = cal_std(data_np_list[i])
                plt.plot(std, label=str(i))
                plt.legend()
                plt.savefig('../caches/std_' + str(i) + '.jpg', bbox_inches='tight')
                plt.close()
        else:
            for i in range(len(data_np_list)):
                std = cal_std(data_np_list[i])
                plt.plot(std, label=str(i))
            plt.legend()
            plt.savefig('../caches/std.jpg', bbox_inches='tight')
            plt.close()

    if 'mean' in name:
        plt.title('各类数据均值分布')
        plt.xlabel('')
        plt.ylabel('均值')
        if flag:
            for i in range(len(data_np_list)):
                mean = cal_mean(data_np_list[i])
                plt.plot(mean, label=str(i))
                plt.legend()
                plt.savefig('../caches/mean_' + str(i) + '.jpg', bbox_inches='tight')
                plt.close()
        else:
            for i in range(len(data_np_list)):
                mean = cal_mean(data_np_list[i])
                plt.plot(mean, label=str(i))
            plt.legend()
            plt.savefig('../caches/mean.jpg', bbox_inches='tight')
            plt.close()

    if 'min' in name:
        plt.title('各类数据最小值分布')
        plt.xlabel('')
        plt.ylabel('最小值')
        if flag:
            for i in range(len(data_np_list)):
                min_ = cal_min(data_np_list[i])
                plt.plot(min_, label=str(i))
                plt.legend()
                plt.savefig('../caches/min_' + str(i) + '.jpg', bbox_inches='tight')
                plt.close()
        else:
            for i in range(len(data_np_list)):
                min_ = cal_min(data_np_list[i])
                plt.plot(min_, label=str(i))
            plt.legend()
            plt.savefig('../caches/min.jpg', bbox_inches='tight')
            plt.close()

    if 'max' in name:
        plt.title('各类数据最大值分布')
        plt.xlabel('')
        plt.ylabel('最大值')
        if flag:
            for i in range(len(data_np_list)):
                max_ = cal_max(data_np_list[i])
                plt.plot(max_, label=str(i))
                plt.legend()
                plt.savefig('../caches/max_' + str(i) + '.jpg', bbox_inches='tight')
                plt.close()
        else:
            for i in range(len(data_np_list)):
                max_ = cal_max(data_np_list[i])
                plt.plot(max_, label=str(i))
            plt.legend()
            plt.savefig('../caches/max.jpg', bbox_inches='tight')
            plt.close()


def scatter_all(mode, flag=False, *name):
    """ plot feature in caches
        :param mode: 'gen' or 'real'
        :param flag: a data a figure when true; all data a figure when false
        :param name: 'std' 'mean' 'min' 'max'
        :return: scatter
    """
    if mode in ('data_selfnoise', 'data_wcgan'):
        data_np_list = read_gen_data(mode)
    elif mode == 'real':
        data_np_list = read_csv_data()
    else:
        raise ValueError('input mode error')

    if 'std' in name:
        plt.title('各类数据标准差分布')
        plt.xlabel('')
        plt.ylabel('标准差')
        if flag:
            for i in range(len(data_np_list)):
                std = cal_std(data_np_list[i])
                x_label = np.ones(std.shape[0]) * i
                plt.scatter(x_label, std, marker='*')
                plt.savefig('../caches/std_' + str(i) + '.jpg', bbox_inches='tight')
                plt.close()
        else:
            for i in range(len(data_np_list)):
                std = cal_std(data_np_list[i])
                x_label = np.ones(std.shape[0]) * i
                plt.scatter(x_label, std, marker='*')
                plt.xticks(np.arange(len(data_np_list)), list(range(len(data_np_list))))
            plt.savefig('../caches/std.jpg', bbox_inches='tight')
            plt.close()

    if 'mean' in name:
        plt.title('各类数据均值分布')
        plt.xlabel('')
        plt.ylabel('均值')
        if flag:
            for i in range(len(data_np_list)):
                mean = cal_mean(data_np_list[i])
                plt.plot(mean, label=str(i))
                plt.legend()
                plt.savefig('../caches/mean_' + str(i) + '.jpg', bbox_inches='tight')
                plt.close()
        else:
            for i in range(len(data_np_list)):
                mean = cal_mean(data_np_list[i])
                x_label = np.ones(mean.shape[0]) * i
                plt.scatter(x_label, mean, marker='*')
                plt.xticks(np.arange(len(data_np_list)), list(range(len(data_np_list))))
            plt.savefig('../caches/mean.jpg', bbox_inches='tight')
            plt.close()

    if 'min' in name:
        plt.title('各类数据最小值分布')
        plt.xlabel('')
        plt.ylabel('最小值')
        if flag:
            for i in range(len(data_np_list)):
                min_ = cal_min(data_np_list[i])
                plt.plot(min_, label=str(i))
                plt.legend()
                plt.savefig('../caches/min_' + str(i) + '.jpg', bbox_inches='tight')
                plt.close()
        else:
            for i in range(len(data_np_list)):
                min_ = cal_min(data_np_list[i])
                x_label = np.ones(min_.shape[0]) * i
                plt.scatter(x_label, min_, marker='*')
                plt.xticks(np.arange(len(data_np_list)), list(range(len(data_np_list))))
            plt.savefig('../caches/min.jpg', bbox_inches='tight')
            plt.close()

    if 'max' in name:
        plt.title('各类数据最大值分布')
        plt.xlabel('')
        plt.ylabel('最大值')
        if flag:
            for i in range(len(data_np_list)):
                max_ = cal_max(data_np_list[i])
                plt.plot(max_, label=str(i))
                plt.legend()
                plt.savefig('../caches/max_' + str(i) + '.jpg', bbox_inches='tight')
                plt.close()
        else:
            for i in range(len(data_np_list)):
                max_ = cal_max(data_np_list[i])
                x_label = np.ones(max_.shape[0]) * i
                plt.scatter(x_label, max_, marker='*')
                plt.xticks(np.arange(len(data_np_list)), list(range(len(data_np_list))))
            plt.savefig('../caches/max.jpg', bbox_inches='tight')
            plt.close()


def cal_corrcoef():
    real_data = read_csv_data()
    fake_data_self = read_gen_data('data_selfnoise')
    fake_data_wcgan = read_gen_data('data_wcgan')
    data_num = 1
    n_class = len(real_data)
    all_coef = np.empty((n_class * data_num * 3, data_num * 3), dtype=float)
    assert len(real_data) == len(fake_data_self)
    assert len(fake_data_self) == len(fake_data_wcgan)
    for i in range(n_class):
        real_data_sort = np.empty((data_num, 1024), dtype=float)
        fake_data_sort_self = np.empty((data_num, 1024), dtype=float)
        fake_data_sort_wcgan = np.empty((data_num, 1024), dtype=float)
        for j in range(data_num):
            index = random.randint(0, real_data[i].shape[0] - 1)
            real_data_sort[j] = real_data[i][index]

            index = random.randint(0, fake_data_self[i].shape[0] - 1)
            fake_data_sort_self[j] = fake_data_self[i][index]

            index = random.randint(0, fake_data_wcgan[i].shape[0] - 1)
            fake_data_sort_wcgan[j] = fake_data_wcgan[i][index]

        matrix = np.concatenate((real_data_sort, fake_data_sort_self, fake_data_sort_wcgan), axis=0)
        coef = np.corrcoef(matrix)
        for k in range(data_num * 3):
            all_coef[i * data_num * 3 + k] = coef[k]

    print(all_coef)
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet('0')
    for i in range(all_coef.shape[0]):
        for j in range(all_coef.shape[1]):
            sheet.write(i + i // (data_num * 3), j, all_coef[i][j])
    workbook.save('corrcoef.xls')
    return 0


def cal_fft():
    col_num = 2
    n_class = 5
    fake_data_selfnoise = read_gen_data('data_selfnoise')
    fake_data_wcgan = read_gen_data('data_wcgan')
    real_data = read_csv_data()
    for i in range(len(fake_data_selfnoise)):
        for j in range(col_num):
            plt.subplot(n_class, col_num, i * col_num + j + 1)
            index = random.randint(0, fake_data_selfnoise[i].shape[0] - 1)
            f, freq = fft(fake_data_selfnoise[i][index])
            plt.plot(freq, f)
    plt.tight_layout()
    # plt.show()
    plt.savefig('../caches/real_f.jpg')
    plt.close()

    for i in range(len(fake_data_wcgan)):
        for j in range(col_num):
            plt.subplot(n_class, col_num, i * col_num + j + 1)
            index = random.randint(0, fake_data_wcgan[i].shape[0] - 1)
            f, freq = fft(fake_data_wcgan[i][index])
            plt.plot(freq, f)
    plt.tight_layout()
    # plt.show()
    plt.savefig('../caches/wcgan_f.jpg')
    plt.close()

    for i in range(len(real_data)):
        for j in range(col_num):
            plt.subplot(n_class, col_num, i * col_num + j + 1)
            index = random.randint(0, real_data[i].shape[0] - 1)
            f, freq = fft(real_data[i][index])
            plt.plot(freq, f)
    plt.tight_layout()
    # plt.show()
    plt.savefig('../caches/self_f.jpg')
    plt.close()

    for i in range(n_class):
        plt.subplot(n_class, 3, i * 3 + 1)
        index = random.randint(0, fake_data_selfnoise[i].shape[0] - 1)
        f, freq = fft(fake_data_selfnoise[i][index])
        plt.plot(freq, f)

        plt.subplot(n_class, 3, i * 3 + 2)
        index = random.randint(0, fake_data_wcgan[i].shape[0] - 1)
        f, freq = fft(fake_data_wcgan[i][index])
        plt.plot(freq, f)

        plt.subplot(n_class, 3, i * 3 + 3)
        index = random.randint(0, real_data[i].shape[0] - 1)
        f, freq = fft(real_data[i][index])
        plt.plot(freq, f)
    plt.tight_layout()
    # plt.show()
    plt.savefig('../caches/all_f.jpg')
    plt.close()


if __name__ == '__main__':
    # scatter_all('data_selfnoise', False, 'std', 'mean', 'min', 'max')
    # cal_corrcoef()
    cal_fft()
