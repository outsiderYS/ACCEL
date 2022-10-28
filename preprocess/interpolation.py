import numpy as np
from matplotlib import pyplot as plt


def same_means(x_axis, y_axis, z_axis, time_axis):
    last_time = 0
    i = 0
    same_count = 0
    same_total_x = 0
    same_total_y = 0
    same_total_z = 0
    result_x = []
    result_y = []
    result_z = []
    time_result = []
    while i < len(time_axis) - 1:
        if last_time == time_axis[i+1]:
            same_count += 1
            same_total_x += x_axis[i]
            same_total_y += y_axis[i]
            same_total_z += z_axis[i]
        else:
            if same_count > 0:
                same_count += 1
                same_total_x += x_axis[i]
                same_total_y += y_axis[i]
                same_total_z += z_axis[i]
                time_result.append(last_time)
                result_x.append(same_total_x / same_count)
                result_y.append(same_total_y / same_count)
                result_z.append(same_total_z / same_count)
                same_count = 0
                same_total_x = 0
                same_total_y = 0
                same_total_z = 0
                last_time = time_axis[i+1]
            else:
                time_result.append(last_time)
                result_x.append(x_axis[i])
                result_y.append(y_axis[i])
                result_z.append(z_axis[i])
                last_time = time_axis[i+1]
        i += 1
    return result_x, result_y, result_z, time_result


def interpolation(x_axis, y_axis, z_axis, time_axis):
    i = 0
    time_result = []
    result_x = []
    result_y = []
    result_z = []
    while i < len(time_axis) - 1:
        time_result.append(time_axis[i])
        result_x.append(x_axis[i])
        result_y.append(y_axis[i])
        result_z.append(z_axis[i])
        time_dec = time_axis[i+1] - time_axis[i]
        if time_dec > 1:
            dec_x = (x_axis[i+1] - x_axis[i]) / time_dec
            dec_y = (y_axis[i + 1] - y_axis[i]) / time_dec
            dec_z = (z_axis[i + 1] - z_axis[i]) / time_dec
            for j in range(time_axis[i], time_axis[i+1] - 1):
                result_x.append(x_axis[i] + dec_x * (j - time_axis[i] + 1))
                result_y.append(y_axis[i] + dec_y * (j - time_axis[i] + 1))
                result_z.append(z_axis[i] + dec_z * (j - time_axis[i] + 1))
                time_result.append(j+1)
        i += 1
    return result_x, result_y, result_z, time_result


