import torch.nn as nn


class basic_block(nn.Module):
    def __init__(self, in_channels, out_channels, residual):
        super(basic_block, self).__init__()
        stride = 1 if in_channels == out_channels else 2
        self.residual = residual

        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.b1 = nn.BatchNorm2d(out_channels)
        self.a1 = nn.ReLU(inplace=True)

        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.b2 = nn.BatchNorm2d(out_channels)

        if residual:
            self.res_c1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            self.res_b1 = nn.BatchNorm2d(out_channels)

        self.a2 = nn.ReLU(inplace=True)

    def forward(self, input):
        residual = input
        x = self.c1(input)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)

        if self.residual:
            residual = self.res_c1(input)
            residual = self.res_b1(residual)

        out = self.a2(y + residual)
        return out


class bottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, residual):
        super(bottleneck, self).__init__()
        stride = 1 if in_channels == out_channels else 2
        self.residual = residual

        self.c1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=stride)
        self.b1 = nn.BatchNorm2d(mid_channels)

        self.c2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.b2 = nn.BatchNorm2d(mid_channels)

        self.c3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1)
        self.b3 = nn.BatchNorm2d(out_channels)

        if residual:
            self.res_c1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            self.res_b1 = nn.BatchNorm2d(out_channels)
        self.a3 = nn.ReLU(inplace=True)

    def forward(self, input):
        residual = input
        x = self.c1(input)
        x = self.b1(x)

        x = self.c2(x)
        x = self.b2(x)

        x = self.c3(x)
        y = self.b3(x)

        if self.residual:
            residual = self.res_c1(input)
            residual = self.res_b1(residual)

        out = self.a3(y + residual)
        return out



class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.filter = 64
        self.layers = [2,2,2,2]

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.filter, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.filter),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_conv = nn.Sequential()

        for layer_id in range(4):
            for block_id in range(self.layers[layer_id]):
                name = "conv" + str(layer_id+2) + "_" + str(block_id)
                if layer_id != 0 and block_id == 0:
                    block = basic_block(in_channels=self.filter//2, out_channels=self.filter, residual=True)
                else:
                    block = basic_block(in_channels=self.filter, out_channels=self.filter, residual=False)

                self.res_conv.add_module(name, block)
            self.filter *= 2

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flaten = nn.Flatten()
        self.fc = nn.Linear(512, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.res_conv(x)
        x = self.pool(x)
        x = self.flaten(x)
        out = self.fc(x)
        return out


class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        self.filter = 64
        self.layers = [3,4,6,3]

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.filter, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.filter),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_conv = nn.Sequential()

        for layer_id in range(4):
            for block_id in range(self.layers[layer_id]):
                name = "conv" + str(layer_id+2) + "_" + str(block_id)
                if layer_id == 0 and block_id == 0:
                    block = bottleneck(in_channels=self.filter, mid_channels=self.filter, out_channels=self.filter*4, residual=True)
                elif layer_id != 0 and block_id == 0:
                    block = bottleneck(in_channels=self.filter*2, mid_channels=self.filter, out_channels=self.filter*4, residual=True)
                else:
                    block = bottleneck(in_channels=self.filter*4, mid_channels=self.filter, out_channels=self.filter*4, residual=False)

                self.res_conv.add_module(name, block)
            self.filter *= 2
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flaten = nn.Flatten()
        self.fc = nn.Linear(2048, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.res_conv(x)
        x = self.pool(x)
        x = self.flaten(x)
        out = self.fc(x)
        return out
