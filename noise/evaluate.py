import os

import numpy as np
import torch
import torch.nn.functional as F
from models import Generator1D, Discriminator
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import trainset
import math as mt
from torchvision.models.densenet import DenseNet
from advGAN import AccelAdv
import models


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


def image_get(accel):
    length = int(accel[2, -1:].item())
    accel = accel[:, 0:length]
    accel = torch.stft(accel, 256, 8, 128, window=torch.hamming_window(128))
    accel = accel[:, :, :, 0] * accel[:, :, :, 0] + accel[:, :, :, 1] * accel[:, :, :, 1]
    accel.sqrt()
    lower_mark = int(len(accel[0]) * (80 / 500))
    high_mark = int(len(accel[0]) * (300 / 500))
    accel = accel[:, lower_mark:high_mark, :]
    max_x = torch.max(accel[0]).item()
    max_y = torch.max(accel[1]).item()
    max_z = torch.max(accel[2]).item()
    mul_coe = 255 / mt.sqrt(max(max_x, max_y, max_z))

    accel = torch.sqrt(accel)
    accel = accel * mul_coe + 0.5
    accel = accel.int()
    return accel


def images_get(accel):
    images = []
    tensor_trans = transforms.Compose([transforms.Resize(size=(224, 224), interpolation=3),
                                       transforms.ToTensor()])
    for i in accel:
        image = image_get(i)
        torch_resize = transforms.Resize(size=(224, 224), interpolation=3)
        tensor_image = torch_resize(image)
        images.append(tensor_image)
    images = torch.stack(images, dim=0)
    return images


path = '../preprocess/ACCELDataset/png/ONEPLUS 9/number_raw'
pretrained_model = '../recognition/model/best.pth.tar'
gen_model = './model/netG_epoch_40.pth'


if __name__ == '__main__':
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
                                             shuffle=None,
                                             num_workers=1,
                                             drop_last=True)
    adv_gen = models.Generator1D(100, 300).to(device)
    adv_gen = nn.DataParallel(adv_gen)
    adv_gen.load_state_dict(torch.load(gen_model, map_location='cpu'))
    adv_gen.eval()

    accuracy_total = 0
    accuracy_pre_total = 0
    mark = 0

    for i, (data_batch, label_batch) in enumerate(dataloader):
        perturbation = torch.randn(size=(32, 100))  # 随机数产生
        perturbation = adv_gen(perturbation)
        perturbation_total = torch.sum(perturbation)

        images = images_get(data_batch)
        images = images.float()

        p_length = len(perturbation[0])
        x_length = len(data_batch[0][0])
        # add a clipping trick
        adv_accel = torch.clamp(perturbation, -0.1, 0.1)
        adv_accel = torch.stack([adv_accel[:, 0:100], adv_accel[:, 100:200], adv_accel[:, 200:300]], dim=1)
        adv_accel = adv_accel.repeat(1, 1, (x_length // int(p_length / 3)) + 1)[:, :, 0:x_length]
        adv_accel[:, :, -1] = 0
        adv_accel = adv_accel + data_batch

        adv_accel = images_get(adv_accel)
        adv_accel = adv_accel.float()

        logits_model = targeted_model(adv_accel)
        logits_model_pre = targeted_model(images)

        probs_model = F.softmax(logits_model, dim=1)
        probs_model_pre = F.softmax(logits_model_pre, dim=1)

        onehot_labels = torch.eye(10, device=device)[label_batch]

        real = torch.sum(onehot_labels * probs_model, dim=1)
        real_pre = torch.sum(onehot_labels * probs_model_pre, dim=1)

        accuracy = (torch.sum(real)/len(data_batch)).item()
        accuracy_total += accuracy

        accuracy_pre = (torch.sum(real_pre)/len(data_batch)).item()
        accuracy_pre_total += accuracy_pre

        mark += 1
        print("batch: %d, accuracy before: %f , accuracy after: %f" % (mark, accuracy_pre, accuracy))
        print("")




