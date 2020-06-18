from torch.utils.data import dataset
import numpy as np
import os
from PIL import Image


class MyDataSet(dataset.Dataset):
    def __init__(self, data):
        '''
        :param data: shouble be a list of array
        '''
        self.labels = []
        self.data_sets = None
        for i in range(len(data)):
            single_data = data[i]
            self.labels = self.labels + [i] * single_data.shape[0]
            # single_data = single_data[:, np.newaxis, :, :]
            if self.data_sets is None:
                self.data_sets = single_data
            else:
                self.data_sets = np.concatenate((self.data_sets, single_data), axis=0)

        self.len = self.data_sets.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data_sets[index], self.labels[index]


# class MyDataSet_Hp(dataset.Dataset):
#     def __init__(self, path, mode, train_num):
#         self.labels = []
#
#     def __len__(self):
#         return 0
#
#     def __getitem__(self, index):
#         return 0


class MyDataSet1D(dataset.Dataset):
    def __init__(self, data):
        super(MyDataSet1D, self).__init__()
        self.labels = []
        self.data_sets = None
        for i in range(len(data)):
            single_data = data[i]
            self.labels = self.labels + [i] * single_data.shape[0]
            # single_data = single_data[:, np.newaxis, :, :]
            if self.data_sets is None:
                self.data_sets = single_data
            else:
                self.data_sets = np.concatenate((self.data_sets, single_data), axis=0)

        self.len = self.data_sets.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data_sets[index], self.labels[index]


class MyDataSetForImgLoad(dataset.Dataset):
    def __init__(self, path, mode, num=(1, 1)):
        '''
        :param path: Performance/imgdatas
        :param mode: train or test
        :param num: [i,j] if train, i is real num,j is gen num; don't need when test; num is epoch, get n_class*i data
        '''
        super(MyDataSetForImgLoad, self).__init__()
        self.mode = mode
        self.num = num
        self.n_class = 5
        if mode == 'train':
            self.gen_path = path + '/train/gen'
            self.real_path = path + '/train/real'
            origin_gen_file_list, origin_real_file_list = os.listdir(self.gen_path), os.listdir(self.real_path)
            self.gen_file_list = []
            self.real_file_list = []
            self.gen_file_labels = []
            self.real_file_labels = []

            for name in origin_real_file_list:
                if int(name[2:-4]) < num[0]:
                    self.real_file_list.append(name)
                    self.real_file_labels.append(int(name[0]))
            for name in origin_gen_file_list:
                if int(name[2:-4]) < num[1]:
                    self.gen_file_list.append(name)
                    self.gen_file_labels.append(int(name[0]))

            self.file_list = self.real_file_list + self.gen_file_list
            self.file_labels = self.real_file_labels + self.gen_file_labels
        else:
            self.test_path = path + '/test'
            self.test_files = os.listdir(self.test_path)
            self.test_labels = []
            for name in self.test_files:
                self.test_labels.append(int(name[0]))

    def __len__(self):
        if self.mode == 'train':
            return len(self.file_labels)
        else:
            return len(self.test_labels)

    def __getitem__(self, index):
        if self.mode == 'train':
            if index < self.num[0] * self.n_class:
                img = Image.open(self.real_path + '/' + self.file_list[index]).convert('L')
            else:
                img = Image.open(self.gen_path + '/' + self.file_list[index]).convert('L')
            return np.array(img), self.file_labels[index]
        else:
            img = Image.open(self.test_path + '/' + self.test_files[index]).convert('L')
            return np.array(img), self.test_labels[index]


def load_data_in_seq(path_list):
    '''
    :param path_list: input is a list  path of .npz
    :return: a list with data array sorted by class
    '''
    data_sets = []
    for path in path_list:
        npz_file = np.load('coedatas/' + path)
        single_data = npz_file['arr_0']
        data_sets.append(single_data)
    return data_sets


def load_data_in_seq_1d(path):
    '''
    :param path_list: input is a list  path of .csv
    :return: a list with 3d-data array sorted by class (class, data_count, data_length)
    '''
    data_sets = []
    path_list = []
    data_length = 1024
    for i in range(len(os.listdir(path)) - 1):
        path_list.append('data/' + str(i) + '.csv')
    for path in path_list:
        single_data = np.loadtxt(path)
        length = len(single_data)
        data_count = length // data_length
        data = single_data[:data_count * data_length]
        data = np.reshape(data, (data_count, data_length))
        data_sets.append(data)
    return data_sets
