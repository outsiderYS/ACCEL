import random
import numpy as np

from preprocess.interpolation import same_means, interpolation
from preprocess.segmentation import segmentation


def random_noise(input_list, input_now):
    length = len(input_list)
    max_input = max(np.absolute(input_list))
    dec = max_input - abs(input_now)
    return random.uniform(-dec, dec)


def first_order_noise(input_list, input_now):
    max_input = max(input_list)
    dec = abs(abs(max_input) - abs(input_now))
    return random.uniform(-dec, dec)


def first_order_mix(input_list):
    length = len(input_list)
    for i in range(101, length):
        input_list[i] = first_order_noise(input_list[i-101:i-1], input_list[i])



if __name__ == "__main__":
    filename = "{}-{}".format("zero", 5)
    f = open("..\ACCELDataset\\raw\\ONEPLUS 9\\number\\{}\\{}.txt".format("zero", filename), encoding="utf-8")

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
    first_order_mix(word_x_list[0])
    first_order_mix(word_y_list[0])
