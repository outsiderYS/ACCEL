from scipy import signal


def filter(input, type, freq):
    b, a = signal.butter(8, 2*freq/1000, type)  # 8表示滤波器的阶数
    filtedData = signal.filtfilt(b, a, input)
    return filtedData