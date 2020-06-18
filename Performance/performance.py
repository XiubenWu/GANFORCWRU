import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse

from torch.utils.data import DataLoader
import torch.nn as nn
import xlwt

from MyModule import ReWrite
from MyModule import MobileNetV3
from MyModule import LetNet5
from MyModule import VGG16

n_class = 5

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--data_length", type=int, default=1024, help="size of the data length")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.8, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--Gchannels", type=int, default=128, help="start_channels_for_G")
parser.add_argument("--n_classes", type=int, default=5, help="num of class of data (labels)")
parser.add_argument("--eps", type=float, default=0.001, help="break value")
opt = parser.parse_args()
print(opt)


def show_data(data):
    data = data.cpu().detach().numpy()
    plt.rcParams['image.cmap'] = 'gray'
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.axis('off')
        plt.contourf(data[i][0])
    plt.show()
    plt.close()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)


class Classify2D(nn.Module):
    def __init__(self):
        super(Classify2D, self).__init__()

        channel1 = 64
        channel2 = 128
        channel3 = 256

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
            nn.Linear(6 * 6 * channel3, 100),
            nn.Linear(100, n_class),
            nn.Softmax(dim=1)
        )

    def forward(self, imgs):
        x = self.conv(imgs)
        x = x.view(x.size(0), -1)
        x = self.l1(x)
        return x


def load_data(dir, path_list):
    '''
    :param path_list: input is a list  path of .npz
    :return: a list with data array sorted by class
    '''
    data_sets = []
    for path in path_list:
        npz_file = np.load(dir + '/' + path)
        single_data = npz_file['arr_0']
        data_sets.append(single_data)
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


def train_function_2d(min_epoch, max_epoch, train_data_loader, test_data_loader,
                      FloatTensor, LongTensor, cuda):
    classify = VGG16.VGG16()
    """
    MobileNetV3.MobileNetV3_Small()
    MobileNetV3.MobileNetV3_Large() ### Gpu out of memory
    Classify2D()
    LetNet5.LeNet5()
    VGG16.VGG16()
    """
    # classify.apply(weights_init)
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
            if epoch > min_epoch and error < opt.eps:
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
            labels = labels.type(LongTensor)

            # a = classify(datas)
            classify_loss = loss(classify(datas), labels)
            classify_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 10 == 0:
            accuracy = 0
            length = 0
            for test_count, (test_data, test_labels) in enumerate(test_data_loader):
                test_data = test_data.view(test_data.shape[0], 1, test_data.shape[-1], test_data.shape[-1])
                # show_data(test_data)
                test_data = test_data.type(FloatTensor)
                class_list = classify(test_data)

                index = np.argmax(class_list.cpu().detach().numpy(), axis=1)
                test_labels = test_labels.detach().numpy()

                for num in range(len(index)):
                    if index[num] == test_labels[num]:
                        accuracy = accuracy + 1
                    # if (index[num] == 0 and test_labels[num] == 0) or (index[num] != 0 and test_labels[num] != 0):
                    #     accuracy = accuracy + 1
                    length = length + 1

                # if (index[num] == 0 and test_labels[num] == 0) or (index[num] != 0 and test_labels[num] != 0):
                #     accuracy = accuracy + 1
            accuracy = accuracy / length
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
    :param data_train: 4-D np
    :param data_test: 4-D np
    :return:
    '''
    # if i % n_class != 0 or j % n_class != 0:
    #     raise ValueError('i or j can be // by n_class')
    os.makedirs('Classify', exist_ok=True)
    para_list = os.listdir('Classify')
    # if 'parameter%d_%d' % (i, j) in para_list:
    #     return -1
    if i == 0 and j == 0:
        return 0.2

    data_test_in = ReWrite.MyDataSet(data_test)

    data_train_in = []
    for count in range(len(data_train)):
        mat1 = data_train[count][0:j]
        mat2 = data_train[count][-(i + 1): -1]
        data_train_in.append(
            np.concatenate((mat1, mat2), axis=0))
    data_train_in = ReWrite.MyDataSet(data_train_in)

    train_data_loader = DataLoader(
        data_train_in,
        batch_size=512,
        shuffle=True
    )
    test_data_loader = DataLoader(
        data_test_in,
        batch_size=512,
        shuffle=True
    )

    cuda = True if torch.cuda.is_available() else False

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    accuracy_ave, classify = train_function_2d(min_epoch, max_epoch, train_data_loader, test_data_loader,
                                               FloatTensor, LongTensor, cuda)

    # torch.save(classify.state_dict(), 'Classify/parameter%d_%d' % (i, j))
    return accuracy_ave


def train_classify(real_num_list, gen_num_list, min_epoch, max_epoch):
    real_data_list = os.listdir('real')
    gen_data_list = os.listdir('gen')
    real_data_set = load_data('real', real_data_list)
    for i in range(len(real_data_set)):
        real_data_set[i] = real_data_set[i][:, np.newaxis, :, :]
    gen_data_set = load_data('gen', gen_data_list)
    data_test = []
    data_train = []
    real_cut = 300  # sort the train data and the test data
    for i in range(len(gen_data_set)):
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
                workbook.save('accuracy.xls')
            except IOError:
                print("write error")
    print(accuracy_np)


def train_function_img(min_epoch, max_epoch, train_data_loader, test_data_loader,
                       FloatTensor, LongTensor, cuda):
    classify = Classify2D()
    classify.apply(weights_init)
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
            if epoch > min_epoch and error < opt.eps:
                break

        for loader_count, (imgs, labels) in enumerate(train_data_loader):
            datas = imgs.view(imgs.shape[0], 1, imgs.shape[-1], imgs.shape[-1])
            # show_data(datas)
            datas = datas.type(FloatTensor)
            labels = labels.type(LongTensor)

            classify_loss = loss(classify(datas), labels)
            classify_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 10 == 0:
            accuracy = 0
            length = 0
            for test_count, (test_data, test_labels) in enumerate(test_data_loader):
                test_data = test_data.view(test_data.shape[0], 1, test_data.shape[-1], test_data.shape[-1])
                # show_data(test_data)
                test_data = test_data.type(FloatTensor)
                class_list = classify(test_data)

                index = np.argmax(class_list.cpu().detach().numpy(), axis=1)
                test_labels = test_labels.detach().numpy()

                for num in range(len(index)):
                    # if index[num] == test_labels[num]:
                    #     accuracy = accuracy + 1
                    if (index[num] == 0 and test_labels[num] == 0) or (index[num] != 0 and test_labels[num] != 0):
                        accuracy = accuracy + 1
                    length = length + 1

                # if (index[num] == 0 and test_labels[num] == 0) or (index[num] != 0 and test_labels[num] != 0):
                #     accuracy = accuracy + 1
            accuracy = accuracy / length
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


def cal_classify_img(i, j, min_epoch, max_epoch):
    os.makedirs('Classify_img', exist_ok=True)
    # para_list = os.listdir('Classify_img')
    # if 'parameter%d_%d' % (i, j) in para_list:
    #     return -1
    if i == 0 and j == 0:
        return 0
    data_train = ReWrite.MyDataSetForImgLoad('imgdatas', mode='train', num=(i, j))
    data_test = ReWrite.MyDataSetForImgLoad('imgdatas', mode='test')

    train_data_loader = DataLoader(
        data_train,
        batch_size=512,
        shuffle=True
    )
    test_data_loader = DataLoader(
        data_test,
        batch_size=1000,
        shuffle=True
    )

    cuda = True if torch.cuda.is_available() else False

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    accuracy_ave, classify = train_function_img(min_epoch, max_epoch, train_data_loader, test_data_loader,
                                                FloatTensor, LongTensor, cuda)

    # torch.save(classify.state_dict(), 'Classify/parameter%d_%d' % (i, j))
    return accuracy_ave


def train_classify_img(real_num_list, gen_num_list, min_epoch, max_epoch):
    accuracy_np = np.empty([len(real_num_list), len(gen_num_list)], dtype=float)
    for i_i in range(len(real_num_list)):
        for j in range(len(gen_num_list)):
            accuracy = cal_classify_img(real_num_list[i_i], gen_num_list[j], min_epoch, max_epoch)
            accuracy_np[i_i][j] = accuracy
            print('real:%d/%d gen:%d/%d' % (i_i, len(real_num_list) - 1, j, len(gen_num_list) - 1))

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
            workbook.save('accuracy_img.xls')
        except IOError:
            print('write error')
    print(accuracy_np)


def main():
    # real_num_list = [0, 10, 100, 500, 1000, 1500]  # can be // by n_class
    # gen_num_list = [0, 10, 100, 500, 1000]  # can be // by n_class
    real_num_list = [1000, 1500]  # can be // by n_class
    # gen_num_list = [0]  # can be // by n_class
    gen_num_list = list(range(0, 1001, 50))
    gen_num_list = [x // n_class for x in gen_num_list]
    real_num_list = [x // n_class for x in real_num_list]
    min_epoch = 800
    max_epoch = 1500

    train_classify(real_num_list, gen_num_list, min_epoch, max_epoch)
    # train_classify_img(real_num_list, gen_num_list, min_epoch, max_epoch)


if __name__ == '__main__':
    main()
