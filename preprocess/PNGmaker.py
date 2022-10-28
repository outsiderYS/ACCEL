import matplotlib.pyplot as plt
import numpy as np
import librosa.core as lc
import librosa.display
import PIL.Image as Image
import math as mt


def spectrogram_list(input_axis, n_fft=1024):
    mag = np.abs(lc.stft(input_axis, n_fft=n_fft, hop_length=2, win_length=64, window='hamming'))
    return mag


def PNGmaker(x_axis, y_axis, z_axis, note, num):
    list_x = spectrogram_list(x_axis, 1024)
    list_y = spectrogram_list(y_axis, 1024)
    list_z = spectrogram_list(z_axis, 1024)
    lower_mark = int(len(list_z) * (80/500))
    high_mark = int(len(list_z) * (300/500))
    list_x = list_x[lower_mark:high_mark, :]
    list_y = list_y[lower_mark:high_mark, :]
    list_z = list_z[lower_mark:high_mark, :]
    max_x = np.max(list_x)
    max_y = np.max(list_y)
    max_z = np.max(list_z)
    mul_coe = 255/mt.sqrt(max(max_x, max_y, max_z))
    row = len(list_z)
    column = len(list_z[0])
    image = Image.new("RGB", (row, column))
    for i in range(row):
        for j in range(column):
            image.putpixel((i, j), (int(mt.sqrt(list_x[i, j])*mul_coe + 0.5),
                           int(mt.sqrt(list_y[i, j])*mul_coe + 0.5), int(mt.sqrt(list_z[i, j])*mul_coe + 0.5)))
    image.save(".\ACCELDataset\\png\\ONEPLUS 9\\number\\{}\\{}.png".format(note, num))


def PNGmakers(x_axis, y_axis, z_axis, note):
    length = len(x_axis)
    for i in range(0, length):
        PNGmaker(x_axis[i], y_axis[i], z_axis[i], note, i)
    print("png图片绘制完成")
