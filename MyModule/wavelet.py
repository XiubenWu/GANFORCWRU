import matplotlib.pyplot as plt
import numpy as np
import pywt


# from matplotlib.font_manager import FontProperties
#
# chinese_font = FontProperties(fname='/usr/share/fonts/SimHei.ttc')

# plt.rcParams['font.sans-serif'] = ['SimHei']
# sampling_rate = 1024
# t = np.arange(0, 1.0, 1.0 / sampling_rate)
# f1 = 100
# f2 = 200
# f3 = 300
# data = np.piecewise(t, [t < 1, t < 0.8, t < 0.3],
#                     [lambda t: np.sin(2 * np.pi * f1 * t), lambda t: np.sin(2 * np.pi * f2 * t),
#                      lambda t: np.sin(2 * np.pi * f3 * t)])


def cal_wave(data, sampling_rate, total_scale=32, wave_name='cgau8'):
    time = np.arange(0, len(data)) / sampling_rate
    fc = pywt.central_frequency(wave_name)
    c_param = 2 * fc * total_scale
    scales = c_param / np.arange(total_scale, 1, -1)
    [cwt_mat, frequencies] = pywt.cwt(data, scales, wave_name, 1.0 / sampling_rate)
    # plt.figure(figsize=(8, 4))
    # plt.subplot(211)
    # plt.plot(time, data)
    # plt.xlabel(u"Time(s)")
    # plt.title(u"300HzAnd200HzAnd100Hz")
    # plt.subplot(212)
    # plt.contourf(time, frequencies, abs(cwt_mat))
    # plt.ylabel(u"Frequency(Hz)")
    # plt.xlabel(u"Time(s)")
    # plt.subplots_adjust(hspace=0.4)
    # plt.show()
    return cwt_mat, frequencies

# 把数据经小波变换返回系数矩阵，无频率信息，可直接用于训练


# cal_wave(data, sampling_rate, 512)
