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


path = '../preprocess/ACCELDataset/png/ONEPLUS 9/number_raw'
pretrained_model = '../recognition/model/best.pth.tar'


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.1)


def spectrogram_list(input_axis, n_fft=1024):
    mag = np.abs(lc.stft(input_axis, n_fft=n_fft, hop_length=8, win_length=128, window='hamming'))
    return mag


def image_get(accel):
    accel = torch.stft(accel, 256, 8, 128, window=torch.hamming_window(128))
    accel = accel[:, :, :, 0] * accel[:, :, :, 0] + accel[:, :, :, 1] * accel[:, :, :, 1]
    accel.sqrt()
    lower_mark = int(len(accel[0]) * (80/500))
    high_mark = int(len(accel[0]) * (300/500))
    accel = accel[:, lower_mark:high_mark, :]
    max_x = torch.max(accel[0]).item()
    max_y = torch.max(accel[1]).item()
    max_z = torch.max(accel[2]).item()
    mul_coe = 255/mt.sqrt(max(max_x, max_y, max_z))

    accel = torch.sqrt(accel)
    accel = accel*mul_coe + 0.5
    accel = accel.int()
    return accel


def images_get(accel):
    images = []
    tensor_trans = transforms.Compose([transforms.Resize(size=(224, 224), interpolation=3),
                                       transforms.ToTensor()])
    for i in accel:
        image = image_get(i)
        # image_arr = np.array(image)
        torch_resize = transforms.Resize(size=(224, 224), interpolation=3)
        tensor_image = torch_resize(image)
        images.append(tensor_image)
    images = torch.stack(images, dim=0)
    return images


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
    gen = Generator1D(100, 300)
    disc = Discriminator()
    dataset = trainset.TrainSet(path=path, Train=False)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=32,
                                            shuffle=True,
                                            num_workers=1,
                                            drop_last=False)

    gen.apply(weights_init)
    disc.apply(weights_init)
    optimizer_G = torch.optim.Adam(gen.parameters(),
                                   lr=0.001)
    optimizer_D = torch.optim.Adam(disc.parameters(),
                                   lr=0.001)
    for i, (data_batch, label_batch) in enumerate(dataloader):
        noise = torch.randn(size=(32, 100))
        perturbation = gen(noise)

        images = images_get(data_batch)
        p_length = len(perturbation[0])
        x_length = len(data_batch[0][0])
        # add a clipping trick
        adv_accel = torch.clamp(perturbation, -0.1, 0.1)
        adv_accel = torch.stack([adv_accel[:, 0:100], adv_accel[:, 100:200], adv_accel[:, 200:300]], dim=1)
        adv_accel = adv_accel.repeat(1, 1, (x_length // int(p_length/3)) + 1)[:, :, 0:x_length]
        adv_accel = adv_accel + data_batch

        images = images.float()
        pred_real = disc.forward(images)

        optimizer_D.zero_grad()
        loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device="cpu"))
        loss_D_real.backward()

        adv_accel = images_get(adv_accel)
        adv_accel = adv_accel.float()
        pred_fake = disc(adv_accel.detach())
        loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device="cpu"))
        loss_D_fake.backward()
        loss_D_GAN = loss_D_fake + loss_D_real
        optimizer_D.step()

        optimizer_G.zero_grad()

        # cal G's loss in GAN
        pred_fake = disc(adv_accel)
        loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device="cpu"))
        loss_G_fake.backward(retain_graph=True)

        # calculate perturbation norm
        C = 0.1
        loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))  # 计算dist
        # loss_perturb = torch.max(loss_perturb - C, torch.zeros(1, device=self.device))

        # cal adv loss
        device = torch.device('cpu')
        targeted_model = DenseNet(growth_rate=32, block_config=(6, 12, 24, 16),
                         num_init_features=64, bn_size=4, drop_rate=0.3, num_classes=10).to(device)
        targeted_model = nn.DataParallel(targeted_model)
        load_checkpoint(pretrained_model, targeted_model)
        targeted_model.eval()

        logits_model = targeted_model(adv_accel)
        probs_model = F.softmax(logits_model, dim=1)
        onehot_labels = torch.eye(10, device=device)[label_batch]

        # C&W loss function
        real = torch.sum(onehot_labels * probs_model, dim=1)
        other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
        zeros = torch.zeros_like(other)
        loss_adv = torch.max(real - other, zeros)
        loss_adv = torch.sum(loss_adv)

        # maximize cross_entropy loss
        # loss_adv = - F.mse_loss(logits_model, onehot_labels)
        # loss_adv = - F.cross_entropy(logits_model, labels)

        adv_lambda = 10
        pert_lambda = 1
        loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb
        loss_G.backward()
        optimizer_G.step()
        print("success")
