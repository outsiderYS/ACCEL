import random
import numpy as np

from preprocess.interpolation import same_means, interpolation
from preprocess.segmentation import segmentation
from preprocess.filter import filter
from preprocess.draw import display_waveform
from mixed_PNGmaker import PNGmakers


def random_noise(input_list, input_now):
    length = len(input_list)
    max_input = max(np.absolute(input_list))
    dec = max_input - abs(input_now)
    return random.uniform(-dec, dec)


def first_order_noise(input_list, input_now):
    max_input = max(input_list)
    dec = abs(abs(max_input) - abs(input_now))
    return random.uniform(-dec, dec)


def first_order_mix(x_list, y_list, z_list):
    length = len(x_list)
    for i in range(0, length):
        word_list_length = len(x_list[i])
        for j in range(101, word_list_length):
            x_list[i][j] = first_order_noise(x_list[i][j - 101:j - 1], x_list[i][j])
            y_list[i][j] = first_order_noise(y_list[i][j - 101:j - 1], y_list[i][j])
            z_list[i][j] = first_order_noise(z_list[i][j - 101:j - 1], z_list[i][j])


def order_mix(marknum, calnum, start_num):
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
    first_order_mix(word_x_list, word_y_list, word_z_list)
    PNGmakers(word_x_list, word_y_list, word_z_list, marknum, start_num)


if __name__ == "__main__":
    num_list = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    for i in num_list:
        for j in range(5, 31):
            order_mix(i, j, (j-1)*100)
            print(i)
            print(j)
