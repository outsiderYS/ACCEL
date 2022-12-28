import matplotlib.pyplot as plt
import numpy as np
import librosa.core as lc
import librosa.display
import PIL.Image as Image
import math as mt

import torchaudio
import torchvision.transforms as transforms

import torch

from filter import filter


def spectrogram_list(input_axis, n_fft=1024):
    mag = np.abs(lc.stft(input_axis, n_fft=n_fft, hop_length=8, win_length=128, window='hamming'))
    return mag


def PNGmaker(x_axis, y_axis, z_axis, note, num):
    f = open(".\ACCELDataset\\png\\ONEPLUS 9\\number_raw\\{}\\{}_x.txt".format(note, num), "w")
    for i in x_axis:
        s = str(i) + "\n"
        f.write(s)
    f.close()
    f = open(".\ACCELDataset\\png\\ONEPLUS 9\\number_raw\\{}\\{}_y.txt".format(note, num), "w")
    for i in y_axis:
        s = str(i) + "\n"
        f.write(s)
    f.close()
    f = open(".\ACCELDataset\\png\\ONEPLUS 9\\number_raw\\{}\\{}_z.txt".format(note, num), "w")
    for i in z_axis:
        s = str(i) + "\n"
        f.write(s)
    f.close()

    wav = [x_axis, y_axis, z_axis]
    wav = torch.tensor(wav)

    transform = torchaudio.transforms.Spectrogram(
        n_fft=256,
        win_length=128,
        hop_length=8,
        center=True,
        pad_mode="reflect",
        power=1.0,
    )

    wav = transform(wav)
    accel = wav

    lower_mark = int(len(accel[0]) * (80 / 500))
    high_mark = int(len(accel[0]) * (300 / 500))
    accel = accel[:, lower_mark:high_mark, :]

    accel = torch.sqrt(accel)

    accel = transforms.ToPILImage()(accel)

    accel.save(".\ACCELDataset\\png\\ONEPLUS 9\\number\\{}\\{}.png".format(note, num))


def PNGmakers(x_axis, y_axis, z_axis, note, start_num):
    length = len(x_axis)
    for i in range(0, length):
        PNGmaker(x_axis[i], y_axis[i], z_axis[i], note, i + start_num)
    print("png图片绘制完成")
