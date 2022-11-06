from interpolation import same_means, interpolation
from draw import display_waveform, display_spectrum, display_smooth, display_word_wave, display_word_spectrum
from draw import display_word_spectrogram
from filter import filter
from pre_smooth import smooth
import matplotlib.pyplot as plt
from segmentation import word_location, segmentation
import librosa.core as lc
import librosa.display
import numpy as np
from PNGmaker import PNGmakers


def preprogress(marknum, calnum, start_num):
    filename = "{}-{}".format(marknum, calnum)
    f = open("..\ACCELDataset\\raw\\ONEPLUS 9\\number\\{}\\{}.txt".format(marknum, filename), encoding="utf-8")

    time_axis = []
    x_axis = []
    y_axis = []
    z_axis = []

    try:
        while True:
            value_s = f.readline()
            if value_s == '':
                break
            values = value_s.split(",")
            time_axis.append(int(values[0]))
            x_axis.append(float(values[1]))
            y_axis.append(float(values[2]))
            z_axis.append(float(values[3]))
    finally:
        f.close()

    time_begin = time_axis[0]

    # 时间归零
    for i in range(len(time_axis)):
        time_axis[i] = time_axis[i] - time_begin

    x_axis, y_axis, z_axis, time_axis = same_means(x_axis, y_axis, z_axis, time_axis)
    x_axis, y_axis, z_axis, time_axis = interpolation(x_axis, y_axis, z_axis, time_axis)
    # 幅度归零
    x_sum_10 = 0
    y_sum_10 = 0
    z_sum_10 = 0
    for i in range(10):
        x_sum_10 += x_axis[i]
        y_sum_10 += y_axis[i]
        z_sum_10 += z_axis[i]
    avarage_x = x_sum_10 / 10
    avarage_y = y_sum_10 / 10
    avarage_z = z_sum_10 / 10
    for i in range(len(x_axis)):
        x_axis[i] = x_axis[i] - avarage_x
        y_axis[i] = y_axis[i] - avarage_y
        z_axis[i] = z_axis[i] - avarage_z

    x_axis_filter = filter(x_axis, 'highpass', 80)
    y_axis_filter = filter(y_axis, 'highpass', 80)
    z_axis_filter = filter(z_axis, 'highpass', 80)

    word_x_list, word_y_list, word_z_list = segmentation(x_axis_filter, y_axis_filter, z_axis_filter)
    #display_word_wave(word_x_list, word_y_list, word_z_list)
    #display_word_spectrum(word_x_list, word_y_list, word_z_list)
    #display_word_spectrogram(word_z_list[0])
    #display_smooth(z_axis)
    PNGmakers(word_x_list, word_y_list, word_z_list, marknum, start_num)


if __name__ == "__main__":
    # num_list = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    # for i in num_list:
    #     preprogress(i, 30, 2900)
    #     print(i)
    preprogress("zero", 30, 2900)

