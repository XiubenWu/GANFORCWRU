import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse

from torch.utils.data import DataLoader

import torch.nn as nn


from MyModule import ReWrite
from MyModule import VGG16

import xlwt

n_class = 5

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--data_length", type=int, default=1024, help="size of the data length")
parser.add_argument("--lr", type=float, default=0.00008, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.8, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--Gchannels", type=int, default=128, help="start_channels_for_G")
parser.add_argument("--n_classes", type=int, default=5, help="num of class of data (labels)")
opt = parser.parse_args()
print(opt)


class Classify1D(nn.Module):
    def __init__(self):
        super(Classify1D, self).__init__()

        channel1 = 16
        channel2 = 32
        channel3 = 64

        self.conv = nn.Sequential(
            nn.Conv1d(1, channel1, 11, 1),
            nn.BatchNorm1d(channel1),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(channel1, channel2, 11, 1),
            nn.BatchNorm1d(channel2),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(channel2, channel3, 11, 1),
            nn.BatchNorm1d(channel3),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(channel3, 1, 11, 1),

            nn.Linear(1024 - 10 * 4, 100),
            nn.Linear(100, n_class)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, datas):
        x = self.conv(datas)
        x = x.view(x.shape[0], x.shape[-1])
        x = self.softmax(x)
        return x


def load_data_1d(dir, path_list):
    '''
    :param path_list: input is a list  path of .npz
    :return: a list with data array sorted by class
    '''
    data_sets = []
    data_length = 1024
    for path in path_list:
        data = np.loadtxt(dir + '/' + path)
        length = len(data)
        data_count = length // data_length
        data = data[:data_count * data_length]
        data = np.reshape(data, (data_count, data_length))

        data_sets.append(data)
    return data_sets


def to_one_hot(labels):
    '''
    :param labels: np
    :return: np shanpe(x,n_class)
    '''
    one_hots = np.zeros([len(labels), n_class], dtype=int)
    for index in range(len(labels)):
        one_hots[index][labels[index]] = 1
    return one_hots


def train_function_1d(min_epoch, max_epoch, train_data_loader, test_data_loader,
                      FloatTensor, LongTensor, cuda, eps=1e-3):
    classify = VGG16.VGG161D()
    """
    Classify1D()
    VGG16.VGG161D()
    """
    loss = torch.nn.CrossEntropyLoss()

    if cuda:
        classify.cuda()
        loss.cuda()

    optimizer = torch.optim.Adam(classify.parameters(), opt.lr, (opt.b1, opt.b2))

    ite = 0
    loss_np = np.zeros(10, dtype=float)
    accuracy_np = np.zeros(10, dtype=float)
    error = 1
    accuracy_max = 0
    for epoch in range(opt.n_epochs):
        if epoch > max_epoch:
            break
        else:
            if epoch > min_epoch and error < eps:
                break

        for loader_count, (datas, labels) in enumerate(train_data_loader):
            # one_hots = to_one_hot(labels.detach().numpy())
            # one_hots = LongTensor(one_hots)

            # for k in range(len(labels)):
            #     if labels.numpy()[k] == 1:
            #         plt.contourf(datas.numpy()[k][0])
            #         plt.show()
            #         plt.close()

            datas = datas.type(FloatTensor)
            datas = datas.view(datas.shape[0], 1, datas.shape[-1])
            labels = labels.type(LongTensor)

            classify_loss = loss(classify(datas), labels)
            classify_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        num_sum = 0
        accuracy = 0
        if epoch % 10 == 0:
            for test_count, (test_data, test_labels) in enumerate(test_data_loader):
                test_data = test_data.type(FloatTensor)
                test_data = test_data.view(test_data.shape[0], 1, test_data.shape[-1])
                class_list = classify(test_data)
                index = np.argmax(class_list.cpu().detach().numpy(), axis=1)
                test_labels = test_labels.detach().numpy()

                for num in range(len(index)):
                    if index[num] == test_labels[num]:
                        accuracy = accuracy + 1
                num_sum = num_sum + len(index)

                # if (index[num] == 0 and test_labels[num] == 0) or (index[num] != 0 and test_labels[num] != 0):
                #     accuracy = accuracy + 1
            accuracy = accuracy / num_sum
            # if accuracy > accuracy_max:
            #     accuracy_max = accuracy
            accuracy_np[ite] = accuracy
            loss_np[ite] = classify_loss.item()
            ite = ite + 1
            if ite == 10:
                ite = 0
            error = np.std(loss_np)
            print(
                '[Epoch %d/%d] [loss: %f,std: %f] [accuracy %f%%]' % (
                    epoch, opt.n_epochs, classify_loss.item(), error, accuracy * 100))
    return np.mean(accuracy_np), classify


def cal_classify(i, j, min_epoch, max_epoch, data_train, data_test):
    '''
    :param i: real data num
    :param j: gen data num
    :param min_epoch: meaning
    :param max_epoch: meaning
    :param data_train: 3-D np
    :param data_test: 3-D np
    :return:
    '''
    # if i % n_class != 0 or j % n_class != 0:
    #     raise ValueError('i or j can be // by n_class')
    os.makedirs('Classify_1d', exist_ok=True)
    para_list = os.listdir('Classify_1d')
    # if 'parameter%d_%d' % (i, j) in para_list:
    #     return -1
    if i == 0 and j == 0:
        return 0.2

    data_test_in = ReWrite.MyDataSet1D(data_test)

    data_train_in = []
    for count in range(len(data_train)):
        mat1 = data_train[count][0:j]
        mat2 = data_train[count][-(i + 1): -1]
        data_train_in.append(
            np.concatenate((mat1, mat2), axis=0))  # mat1:gen mat2:real
    data_train_in = ReWrite.MyDataSet1D(data_train_in)

    train_data_loader = DataLoader(
        data_train_in,
        batch_size=256,
        shuffle=True
    )
    test_data_loader = DataLoader(
        data_test_in,
        batch_size=256,
        shuffle=True
    )

    cuda = True if torch.cuda.is_available() else False

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    accuracy_ave, classify = train_function_1d(min_epoch, max_epoch, train_data_loader, test_data_loader,
                                               FloatTensor, LongTensor, cuda, eps=5e-3)

    # torch.save(classify.state_dict(), 'Classify/parameter%d_%d' % (i, j))
    return accuracy_ave


def train_classify(real_num_list, gen_num_list, min_epoch, max_epoch):
    real_data_list = os.listdir('real_1d')
    real_data_set = load_data_1d('real_1d', real_data_list)
    gen_data_set = np.load("gen_1d/data_wcgan.npz")['arr_0']
    """
    data_selfnoise
    data_wcgan
    """
    data_test = []
    data_train = []
    real_cut = 300  # sort the train data and the test data
    for i in range(gen_data_set.shape[0]):
        mid = np.concatenate((gen_data_set[i], real_data_set[i][0:real_cut]), axis=0)
        data_train.append(mid)
        data_test.append(real_data_set[i][real_cut:-1])

    accuracy_np = np.empty([len(real_num_list), len(gen_num_list)], dtype=float)
    for i_i in range(len(real_num_list)):
        for j in range(len(gen_num_list)):
            print('real:%d/%d gen:%d/%d' % (i_i, len(real_num_list) - 1, j, len(gen_num_list) - 1))
            accuracy = cal_classify(real_num_list[i_i], gen_num_list[j], min_epoch, max_epoch, data_train, data_test)
            accuracy_np[i_i][j] = accuracy

            # save accuracy
            try:
                workbook = xlwt.Workbook()
                sheet = workbook.add_sheet('accuracy')
                for i in range(len(real_num_list)):
                    sheet.write(i + 1, 0, real_num_list[i] * n_class)
                for i in range(len(gen_num_list)):
                    sheet.write(0, i + 1, gen_num_list[i] * n_class)
                for i in range(accuracy_np.shape[0]):
                    for j in range(accuracy_np.shape[1]):
                        sheet.write(i + 1, j + 1, accuracy_np[i][j])
                workbook.save('accuracy_1d.xls')
            except IOError:
                print("write error")
    print(accuracy_np)


def main():
    real_num_list = [0, 10, 100, 500, 1000, 1500]  # can be // by n_class
    gen_num_list = list(range(0, 1001, 50))
    # real_num_list = [1500]  # can be // by n_class
    # gen_num_list = [0]  # can be // by n_class
    gen_num_list = [x // n_class for x in gen_num_list]
    real_num_list = [x // n_class for x in real_num_list]
    min_epoch = 200
    max_epoch = 800
    train_classify(real_num_list, gen_num_list, min_epoch, max_epoch)


if __name__ == '__main__':
    main()
