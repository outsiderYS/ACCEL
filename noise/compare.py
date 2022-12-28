import torch
from PIL import Image
import numpy as np
from torchvision.models import DenseNet
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import math as mt
import matplotlib.pyplot as plt
import torchaudio
import librosa.core as lc


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """

    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint)
    else:
        # this helps avoid errors when loading single-GPU-trained weights onto CPU-model
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


def spectrogram_list(input_axis, n_fft=1024):
    mag = np.abs(lc.stft(input_axis, n_fft=n_fft, hop_length=8, win_length=128, window='hamming'))
    return mag


pretrained_model = '../recognition/model/best.pth.tar'
image = Image.open("../preprocess/ACCELDataset/png/ONEPLUS 9/testdl/eight/1001.png")

img_data = np.array(image)


file_x = open("../preprocess/ACCELDataset/png/ONEPLUS 9/number_raw/eight/1001_x.txt", 'r')
file_y = open("../preprocess/ACCELDataset/png/ONEPLUS 9/number_raw/eight/1001_y.txt", 'r')
file_z = open("../preprocess/ACCELDataset/png/ONEPLUS 9/number_raw/eight/1001_z.txt", 'r')
data_x = file_x.readlines()
data_y = file_y.readlines()
data_z = file_z.readlines()
data_x = list(map(float, data_x))
data_y = list(map(float, data_y))
data_z = list(map(float, data_z))
wav = [data_x, data_y, data_z]
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



max_x = torch.max(accel[0]).item()
max_y = torch.max(accel[1]).item()
max_z = torch.max(accel[2]).item()
mul_coe = 255 / mt.sqrt(max(max_x, max_y, max_z))

accel = torch.sqrt(accel)

accel = transforms.ToPILImage()(accel)

accel.save("fig.png")

plt.imshow(accel)
plt.savefig("fig.png")
plt.show()

print("stop")
