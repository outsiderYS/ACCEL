import torch.utils.data as data
import glob
import os
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch

import imghdr

labels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
testdlpath = '../preprocess/ACCELDataset/png/ONEPLUS 9/testdl.txt'


def wavname_get(rootdir, isTrain):
    wavfiles = []
    testwaves = []
    testfile = open(testdlpath, 'r')
    file_data = testfile.readlines()
    for row in file_data:
        temp_list = row.split('\\')
        temp_num = temp_list[2].split('.')[0]
        temp_name = [temp_list[1], temp_num]
        testwaves.append(temp_name)
    testfile.close()
    if not isTrain:
        return testwaves

    for label in labels:
        path = rootdir + '/' + label
        count = 0
        wavfile_part = []
        for file in os.listdir(path):
            if count == 3:
                count = 1
                wavfile = [label, wavfile_part[0].split('_')[0]]
                wavfiles.append(wavfile)
                wavfile_part = [file]
            else:
                count += 1
                wavfile_part.append(file)
        wavfile = [label, wavfile_part[0].split('_')[0]]
        wavfiles.append(wavfile)
    for i in testwaves:
        r = wavfiles.count(i)
        if r != 0:
            wavfiles.remove(i)
    return wavfiles


def label_switch(label):
    dict = {'zero': 0, 'one': 1, 'two': 2, 'three': 3,
            'four': 4, 'five': 5, 'six': 6, 'seven': 7,
            'eight': 8, 'nine': 9}
    return dict[label]


def wav_get(wavnames, rootdir):
    wavs = []
    max_length = 0
    for i in wavnames:
        filename_x = rootdir + '/' + i[0] + '/' + i[1] + '_x.txt'
        filename_y = rootdir + '/' + i[0] + '/' + i[1] + '_y.txt'
        filename_z = rootdir + '/' + i[0] + '/' + i[1] + '_z.txt'
        file_x = open(filename_x, 'r')
        file_y = open(filename_y, 'r')
        file_z = open(filename_z, 'r')
        data_x = file_x.readlines()
        data_y = file_y.readlines()
        data_z = file_z.readlines()
        data_x = list(map(float, data_x))
        data_y = list(map(float, data_y))
        data_z = list(map(float, data_z))
        length = len(data_x)
        if length > max_length:
            max_length = length
        wav = [[data_x, data_y, data_z], label_switch(i[0])]
        wavs.append(wav)
    for i in wavs:
        ac_len = len(i[0][0])
        i[0][0] += [0.0] * (max_length - ac_len)
        i[0][1] += [0.0] * (max_length - ac_len)
        i[0][2] += [0.0] * (max_length - ac_len)
        i[0][2][-1] = float(ac_len)
    return wavs


class TrainSet(data.Dataset):
    def __init__(self, path, Train=True):
        wavnames = wavname_get(path, Train)
        wavs = wav_get(wavnames, path)
        self.dataset = wavs
        self.Train = Train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        wav, label = self.dataset[idx]
        wav = np.array(wav)
        wav = torch.Tensor(wav)
        return wav, label
