import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import fft
import numpy as np
from scipy import signal
from pre_smooth import smooth
from segmentation import word_location
from filter import filter
import librosa.core as lc
import librosa.display
from PIL import Image


def display_waveform(time_axis, x_axis, y_axis, z_axis):
    plt.figure(dpi=600, figsize=(20, 5))
    plt.plot(time_axis, x_axis)
    plt.savefig("./time/x_axis")
    plt.figure(dpi=600, figsize=(20, 5))
    plt.plot(time_axis, y_axis)
    plt.savefig("./time/y_axis")
    plt.figure(dpi=600, figsize=(20, 5))
    plt.plot(time_axis, z_axis)
    plt.savefig("./time/z_axis")
    print("波形图绘制成功")


def display_spectrum(time_axis, x_axis, y_axis, z_axis):

    ft = fft(x_axis)
    L = len(time_axis)
    p2 = np.absolute(ft)
    p1 = p2[:int(L/2)]
    f = np.arange(int(L/2))*1000/L

    plt.figure(dpi=600, figsize=(20, 5))
    plt.plot(f, 2*p1/L)
    plt.title('Single-Sided Amplitude Spectrum of X(t)')
    plt.xlabel('f (Hz)')
    plt.ylabel('|P1(f)|')
    plt.savefig("./freq/x_axis")

    ft = fft(y_axis)
    p2 = np.absolute(ft)
    p1 = p2[:int(L / 2)]
    f = np.arange(int(L / 2)) * 1000 / L
    plt.figure(dpi=600, figsize=(20, 5))
    plt.plot(f, 2 * p1 / L)
    plt.title('Single-Sided Amplitude Spectrum of Y(t)')
    plt.xlabel('f (Hz)')
    plt.ylabel('|P2(f)|')
    plt.savefig("./freq/y_axis")

    ft = fft(z_axis)
    p2 = np.absolute(ft)
    p1 = p2[:int(L / 2)]
    f = np.arange(int(L / 2)) * 1000 / L
    plt.figure(dpi=600, figsize=(20, 5))
    plt.plot(f, 2 * p1 / L)
    plt.title('Single-Sided Amplitude Spectrum of Z(t)')
    plt.xlabel('f (Hz)')
    plt.ylabel('|P3(f)|')
    plt.savefig("./freq/z_axis")
    print("频谱图绘制成功")


def display_smooth(z_axis):
    z_axis_filter = filter(z_axis, 'highpass', 160)
    list_mark = word_location(z_axis_filter)
    z_axis_200 = smooth(200, z_axis_filter)
    z_axis_200_30 = smooth(30, z_axis_200)
    plt.figure(dpi=600, figsize=(20, 5))
    smooth_time = list(range(0, len(z_axis_200_30)))
    for i in list_mark:
        for j in i:
            plt.text(j, z_axis_200_30[j], '×', color="r")
    plt.plot(smooth_time, z_axis_200_30)
    plt.savefig("./smooth/z_axis")
    print("平滑波形图绘制成功")


def display_word_wave(x_axis_list, y_axis_list, z_axis_list):

    for num in range(0, len(x_axis_list)):
        time_list = list(range(0, len(x_axis_list[num])))

        plt.figure(dpi=600, figsize=(5, 5))
        plt.plot(time_list, x_axis_list[num])
        plt.savefig("./word_time/{}_x_axis".format(num))
        plt.figure(dpi=600, figsize=(5, 5))
        plt.plot(time_list, y_axis_list[num])
        plt.savefig("./word_time/{}_y_axis".format(num))
        plt.figure(dpi=600, figsize=(5, 5))
        plt.plot(time_list, z_axis_list[num])
        plt.savefig("./word_time/{}_z_axis".format(num))

    print("单词波形图绘制成功")


def display_word_spectrum(x_axis_list, y_axis_list, z_axis_list):
    for num in range(0, len(x_axis_list)):

        L = len(x_axis_list[num])
        ft = fft(x_axis_list[num])
        p2 = np.absolute(ft)
        p1 = p2[:int(L / 2)]
        f = np.arange(int(L / 2)) * 1000 / L

        plt.figure(dpi=600, figsize=(5, 5))
        plt.plot(f, 2 * p1 / L)
        plt.title('Single-Sided Amplitude Spectrum of X(t)')
        plt.xlabel('f (Hz)')
        plt.ylabel('|P1(f)|')
        plt.savefig("./word_freq/{}_x_axis".format(num))

        ft = fft(y_axis_list[num])
        p2 = np.absolute(ft)
        p1 = p2[:int(L / 2)]
        f = np.arange(int(L / 2)) * 1000 / L
        plt.figure(dpi=600, figsize=(5, 5))
        plt.plot(f, 2 * p1 / L)
        plt.title('Single-Sided Amplitude Spectrum of Y(t)')
        plt.xlabel('f (Hz)')
        plt.ylabel('|P2(f)|')
        plt.savefig("./word_freq/{}_y_axis".format(num))

        ft = fft(z_axis_list[num])
        p2 = np.absolute(ft)
        p1 = p2[:int(L / 2)]
        f = np.arange(int(L / 2)) * 1000 / L
        plt.figure(dpi=600, figsize=(5, 5))
        plt.plot(f, 2 * p1 / L)
        plt.title('Single-Sided Amplitude Spectrum of Z(t)')
        plt.xlabel('f (Hz)')
        plt.ylabel('|P3(f)|')
        plt.savefig("./word_freq/{}_z_axis".format(num))

    print("单词频谱图绘制成功")


def display_word_spectrogram(input_axis):
    fs = 1000
    n_fft = 1024
    plt.figure(dpi=600, figsize=(10, 5))
    mag = np.abs(lc.stft(input_axis, n_fft=n_fft, hop_length=8, win_length=128, window='hamming'))
    D = librosa.amplitude_to_db(mag, ref=np.max)
    librosa.display.specshow(D, sr=fs, hop_length=8, x_axis='ms', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('broadband spectrogram')
    plt.savefig('./spectrogram/broader.png')
    print("声谱图生成完成")

