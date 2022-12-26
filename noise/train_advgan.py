import os

import numpy as np
import torch
import torch.nn.functional as F
from models import Generator1D, Discriminator
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import trainset
import librosa.core as lc
import math as mt
from torchvision.models.densenet import DenseNet
from preprocess.filter import filter
from advGAN import AccelAdv


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


if __name__ == '__main__':
    path = '../preprocess/ACCELDataset/png/ONEPLUS 9/number_raw'
    pretrained_model = '../recognition/model/best.pth.tar'

    device = torch.device('cpu')
    model_num_labels = 10
    epochs = 50

    targeted_model = DenseNet(growth_rate=32, block_config=(6, 12, 24, 16),
                              num_init_features=64, bn_size=4, drop_rate=0.3, num_classes=model_num_labels).to(device)
    targeted_model = nn.DataParallel(targeted_model)
    load_checkpoint(pretrained_model, targeted_model)
    targeted_model.eval()

    dataset = trainset.TrainSet(path=path, Train=False)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=32,
                                             shuffle=True,
                                             num_workers=1,
                                             drop_last=False)

    accelAdv = AccelAdv(device, targeted_model, model_num_labels, 3, -0.5, 0.5)

    accelAdv.train(dataloader, 50)



