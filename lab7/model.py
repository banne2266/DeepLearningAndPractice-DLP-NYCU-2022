import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_size = 100, condition_size = 100):
        super(Generator, self).__init__()
        self.latent_size = latent_size
        self.condition_size = condition_size

        self.condition_net = nn.Sequential(
            nn.Linear(24, condition_size),
            nn.ReLU()
        )

        channels=[latent_size + condition_size, 512, 256, 128, 64]
        paddings=[(0,0), (1,1), (1,1), (1,1)]
        self.generate_net = nn.Sequential()

        for i in range(1, len(channels)):
            name = "deconv_" + str(i)
            layer = self._make_layer(channels[i-1], channels[i], paddings[i-1])
            self.generate_net.add_module(name, layer)

        self.deconv_5 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
            nn.Tanh()
        )
    
    def forward(self, latent, condition):
        """
        :param latent: (batch_size,100) tensor
        :param condition: (batch_size,24) tensor
        :return: (batch_size,3,64,64) tensor
        """
        latent = latent.view(-1, self.latent_size, 1, 1)
        condition = self.condition_net(condition).view(-1, self.condition_size, 1, 1)
        out = torch.cat((latent, condition), 1)

        out = self.generate_net(out)
        out = self.deconv_5(out)
        return out

    def _make_layer(self, in_channel, out_channel, padding):
        layer = nn.Sequential(
            nn.ConvTranspose2d(in_channel,out_channel,kernel_size=(4,4),stride=(2,2),padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        return layer


class Discriminator(nn.Module):
    def __init__(self, img_shape = (64, 64, 3), is_wgan = False):
        super(Discriminator, self).__init__()

        self.H, self.W, self.C = img_shape

        self.condition_net = nn.Sequential(
            nn.Linear(24, self.C * self.H * self.W),
            nn.LeakyReLU()
        )

        channels=[6, 64, 128, 256, 512]
        self.discriminate_net = nn.Sequential()

        for i in range(1, len(channels)):
            name = "conv_" + str(i)
            layer = self._make_layer(channels[i-1], channels[i])
            self.discriminate_net.add_module(name, layer)

        if is_wgan:
            self.conv_5 = nn.Sequential(
                nn.Conv2d(512, 1, kernel_size = (4,4), stride = (1,1))
            )
        else:
            self.conv_5 = nn.Sequential(
                nn.Conv2d(512, 1, kernel_size = (4,4), stride = (1,1)),
                nn.Sigmoid()
            )

    
    def forward(self, img, condition):
        condition = self.condition_net(condition).view(-1, self.C, self.H, self.W)
        out = torch.cat((img, condition), 1)
        out = self.discriminate_net(out)
        out = self.conv_5(out).view(-1)

        return out

    def _make_layer(self, in_channel, out_channel):
        layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        return layer


class Discriminator_ACGAN(nn.Module):
    def __init__(self, img_shape = (64, 64, 3), is_wgan = False):
        super(Discriminator_ACGAN, self).__init__()
        self.H, self.W, self.C=img_shape
        channels=[3, 64, 128, 256, 512]
        self.discriminate_net = nn.Sequential()

        for i in range(1, len(channels)):
            name = "conv_" + str(i)
            layer = self._make_layer(channels[i-1], channels[i])
            self.discriminate_net.add_module(name, layer)

        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size = (4,4), stride = (1,1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )

        self.flatten = nn.Flatten()

        if is_wgan:
            self.a_net = nn.Sequential(
                nn.Linear(512, 1)
            )
            self.c_net = nn.Sequential(
                nn.Linear(512, 24)
            )
        else:
            self.a_net = nn.Sequential(
                nn.Linear(512, 1),
                nn.Sigmoid()
            )
            self.c_net = nn.Sequential(
                nn.Linear(512, 24),
                nn.Sigmoid()
            )
    
    def forward(self, img):
        out = self.discriminate_net(img)
        out = self.conv_5(out)
        out = self.flatten(out)

        adversial = self.a_net(out).view(-1)
        classify = self.c_net(out)

        return adversial, classify

    def _make_layer(self, in_channel, out_channel):
        layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True)
        )
        return layer


