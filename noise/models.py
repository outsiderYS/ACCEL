import math

import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, image_channel=3):
        super(Discriminator, self).__init__()
        # AccelEve: 3*224*224
        model = [
            nn.Conv2d(image_channel, 24, kernel_size=16, stride=4, padding=0, bias=True),
            nn.LeakyReLU(0.2),
            # 24*53*53
            nn.Conv2d(24, 48, kernel_size=16, stride=4, padding=0, bias=True),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2),
            # 48*10*10
            nn.Conv2d(48, 96, kernel_size=8, stride=4, padding=0, bias=True),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),
            # 96*1*1
            nn.Conv2d(96, 1, 1),
            nn.Sigmoid()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        output = self.model(x).squeeze()
        return output


class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel=None, factor=2, activate=True):
        super(UpBlock, self).__init__()
        self.activate = activate
        out_channel = out_channel if out_channel is not None else in_channel
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=factor, mode='linear', align_corners=True),#factor放大倍数
            nn.Conv1d(in_channel, out_channel, 3,1,2, dilation=2, bias=False),
            nn.BatchNorm1d(out_channel),
        )

    def zero_init(self):
        if self.block[1].weight is not None:
            # nn.init.normal_(self.block[1].weight, mean=0, std=0.000001)
            nn.init.constant_(self.block[1].weight, val=0)
        if self.block[1].bias is not None:
            # nn.init.normal_(self.block[1].bias, mean=0, std=0.000001)
            nn.init.constant_(self.block[1].bias, val=0)

    def forward(self, x):
        x = self.block(x)
        if self.activate:
            x = F.relu(x, inplace=True)
        return x


class InterpolationBlock(nn.Module):
    def __init__(self, in_channel,  output_dim, out_channel=None, activate=True):
        super(InterpolationBlock, self).__init__()
        self.activate = activate
        out_channel = out_channel if out_channel is not None else in_channel
        self.block = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 3,1,2, dilation=2, bias=False),
            nn.BatchNorm1d(out_channel),
        )
        self.output_dim = output_dim

    def zero_init(self):
        if self.block[1].weight is not None:
            # nn.init.normal_(self.block[1].weight, mean=0, std=0.000001)
            nn.init.constant_(self.block[1].weight, val=0)
        if self.block[1].bias is not None:
            # nn.init.normal_(self.block[1].bias, mean=0, std=0.000001)
            nn.init.constant_(self.block[1].bias, val=0)

    def forward(self, x):
        x = F.interpolate(x, self.output_dim, mode='linear')
        x = self.block(x)
        if self.activate:
            x = F.relu(x, inplace=True)
        return x


class Generator1D(nn.Module):
    def __init__(self, noise_dim=100, output_dim=3200):
        super(Generator1D, self).__init__()
        self.noise_dim, self.output_dim = noise_dim, output_dim
        self.base_channel, self.base_size = 64, 25
        self.transform_layer = nn.Linear(noise_dim, self.base_channel * self.base_size, bias=True)
        self.blocks = self.get_blocks(self.base_channel, self.base_size, output_dim)
        self.out_layer = nn.Conv1d(self.base_channel, 1, 3, 1, 1, bias=False)
        self.last_zero_init()

    def last_zero_init(self):
        if self.out_layer.weight is not None:
            nn.init.constant_(self.out_layer.weight, val=0)
        if self.out_layer.bias is not None:
            nn.init.constant_(self.out_layer.bias, val=0)
        # self.blocks[-1].zero_init()

    def get_blocks(self, base_channel, base_size, output_dim):
        layers = int(math.log2(output_dim // base_size))  # 取整除
        blocks = []
        in_dim = base_size
        channel = base_channel
        for idx in range(layers):  # 循环采样
            blocks.append(UpBlock(in_channel=channel, out_channel=channel, activate=True))
            in_dim *= 2
        if not in_dim == output_dim:  # 插值
            blocks.append(
                InterpolationBlock(in_channel=channel, output_dim=output_dim, out_channel=channel, activate=True))

        return nn.Sequential(*blocks)  # 接受多个参数

    def forward(self, x):
        # x created from torch.randn(size=(batch_size, noise_size))
        B = x.shape[0]
        x = self.transform_layer(x)
        x = x.view(B, self.base_channel, self.base_size)
        x = self.blocks(x)
        x = self.out_layer(x)
        #x = self.blocks(x)
        return x.view(B, -1)


# class Generator(nn.Module):
#     def __init__(self,
#                  gen_input_nc,
#                  image_nc,
#                  ):
#         super(Generator, self).__init__()
#
#         encoder_lis = [
#             # MNIST:1*28*28 # AccelEve: 1*(noise_length*3)
#             nn.Conv1d(gen_input_nc, 8, kernel_size=9, stride=2, padding=0, bias=True),
#             nn.InstanceNorm1d(8),
#             nn.ReLU(),
#             # 8*26*26 # 8*(length-9)/2+1
#             nn.Conv1d(8, 16, kernel_size=9, stride=4, padding=0, bias=True),
#             nn.InstanceNorm1d(16),
#             nn.ReLU(),
#             # 16*12*12 # 16*(length-9)/8-1
#             nn.Conv1d(16, 32, kernel_size=9, stride=4, padding=0, bias=True),
#             nn.InstanceNorm1d(32),
#             nn.ReLU(),
#             # 32*5*5 # 32*(length-9)/32-3/2
#         ]
#
#         # bottle_neck_lis = [ResnetBlock(32),
#         #                ResnetBlock(32),
#         #                ResnetBlock(32),
#         #                ResnetBlock(32),]
#
#         decoder_lis = [
#             nn.ConvTranspose1d(32, 16, kernel_size=9, stride=4, padding=0, bias=False),
#             nn.InstanceNorm1d(16),
#             nn.ReLU(),
#             # state size.
#             nn.ConvTranspose1d(16, 8, kernel_size=9, stride=4, padding=0, bias=False),
#             nn.InstanceNorm1d(8),
#             nn.ReLU(),
#             # state size.
#             nn.ConvTranspose1d(8, image_nc, kernel_size=9, stride=2, padding=0, bias=False),
#             nn.Tanh()
#             # state size.
#         ]
#
#         self.encoder = nn.Sequential(*encoder_lis)
#         #self.bottle_neck = nn.Sequential(*bottle_neck_lis)
#         self.decoder = nn.Sequential(*decoder_lis)
#
#     def forward(self, x):
#         x = self.encoder(x)
#         #x = self.bottle_neck(x)
#         x = self.decoder(x)
#         return x
