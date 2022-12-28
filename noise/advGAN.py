import torch.nn as nn
import torch
import numpy as np
import models
import torch.nn.functional as F
import torchvision
import os
import PIL.Image as Image
import math as mt
import torchvision.transforms as transforms

models_path = '../recognition/model'
gen_path = './model/'


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.1)


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
        torch_resize = transforms.Resize(size=(224, 224), interpolation=3)
        tensor_image = torch_resize(image)
        images.append(tensor_image)
    images = torch.stack(images, dim=0)
    return images


class AccelAdv:
    def __init__(self,
                 device,
                 model,
                 model_num_labels,  # 分类数
                 image_nc,  # 图片通道数
                 box_min,   # 信号限制
                 box_max):
        output_nc = image_nc
        self.device = device
        self.model_num_labels = model_num_labels
        self.model = model
        self.input_nc = image_nc
        self.output_nc = output_nc
        self.box_min = box_min
        self.box_max = box_max

        self.gen_input_nc = image_nc  # 生成器输入通道数
        self.netG = models.Generator1D(100, 300).to(device)    # 生成模型
        self.netDisc = models.Discriminator(image_nc).to(device)    # 判别模型

        # initialize all weights
        self.netG.apply(weights_init)   # 初始化参数
        self.netDisc.apply(weights_init)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=0.001)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                            lr=0.001)

        if not os.path.exists(models_path):
            os.makedirs(models_path)

    def train_batch(self, x, labels):   # 训练一批数据
        # optimize D
        for i in range(1):
            #perturbation = self.netG(x)
            perturbation = torch.randn(size=(32, 100))  # 随机数产生
            perturbation = self.netG(perturbation)  # 前向传播生成一批噪声
            p_length = len(perturbation[0])    #
            x_length = len(x[0][0])
            # add a clipping trick
            adv_accel = torch.clamp(perturbation, -0.3, 0.3)  # clamp 数据夹紧到区间
            adv_accel = torch.stack([adv_accel[:, 0:100], adv_accel[:, 100:200], adv_accel[:, 200:300]], dim=1)  # 噪声分割
            adv_accel = adv_accel.repeat(1, 1, (x_length // int(p_length/3)) + 1)[:, :, 0:x_length]  # 添加噪声
            adv_accel = adv_accel + x
            adv_accel = torch.clamp(adv_accel, self.box_min, self.box_max)  # 限制噪声幅度

            self.optimizer_D.zero_grad()    # 最后一层归零
            x = images_get(x)
            x = x.float()
            pred_real = self.netDisc(x)
            loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
            loss_D_real.backward()

            adv_accel = images_get(adv_accel)
            adv_accel = adv_accel.float()
            pred_fake = self.netDisc(adv_accel.detach())
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
            loss_D_fake.backward()
            loss_D_GAN = loss_D_fake + loss_D_real
            self.optimizer_D.step()

        # optimize G
        for i in range(1):
            self.optimizer_G.zero_grad()

            # cal G's loss in GAN
            pred_fake = self.netDisc(adv_accel)
            loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
            loss_G_fake.backward(retain_graph=True)

            # calculate perturbation norm
            C = 0.1
            loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))   # 计算dist
            # loss_perturb = torch.max(loss_perturb - C, torch.zeros(1, device=self.device))

            # cal adv loss
            logits_model = self.model(adv_accel)

            probs_model = F.softmax(logits_model, dim=1)
            onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]

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
            self.optimizer_G.step()

        return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), loss_adv.item()

    def train(self, train_dataloader, epochs):
        for epoch in range(1, epochs+1):

            if epoch == 50:     # 调整学习率
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.0001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.0001)
            if epoch == 80:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.00001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.00001)
            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            for i, data in enumerate(train_dataloader, start=0):
                accel, labels = data
                accel, labels = accel.to(self.device), labels.to(self.device)
                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch = \
                    self.train_batch(accel, labels)
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch
                print("trained a batch")

            # print statistics
            num_batch = len(train_dataloader)
            print("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f,\
             \nloss_perturb: %.3f, loss_adv: %.3f, \n" %
                  (epoch, loss_D_sum/num_batch, loss_G_fake_sum/num_batch,
                   loss_perturb_sum/num_batch, loss_adv_sum/num_batch))

            # save generator
            if epoch%20==0:
                netG_file_name = gen_path + 'netG_epoch_' + str(epoch) + '.pth'
                torch.save(self.netG.state_dict(), netG_file_name)

