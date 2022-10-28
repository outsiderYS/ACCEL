from smooth import smooth
from filter import filter


def word_location(z_axis_filter):
    z_axis_200 = smooth(200, z_axis_filter)
    z_axis_200_30 = smooth(30, z_axis_200)
    max_mag = max(z_axis_200_30)
    min_mag = min(z_axis_200_30)
    threshold = 0.15*max_mag + 0.8*min_mag
    flag = False
    mark = 0
    result = []
    couple = []
    for i in range(0, len(z_axis_200_30) - 200):
        if z_axis_200_30[i] >= threshold and not flag:
            flag = True
            mark = i
            couple.append(i - 100)
        elif z_axis_200_30[i] <= threshold and flag:
            if i - mark > 200:
                couple.append(i + 200)
                result.append(couple)
                couple = []
                flag = False
    return result


def segmentation(x_axis, y_axis, z_axis):
    z_filter = filter(z_axis, "highpass", 160)
    mark_list = word_location(z_filter)
    result_x = []
    result_y = []
    result_z = []
    for i in mark_list:
        result_x.append(x_axis[i[0]:i[1]])
        result_y.append(y_axis[i[0]:i[1]])
        result_z.append(z_axis[i[0]:i[1]])
    return result_x, result_y, result_z
