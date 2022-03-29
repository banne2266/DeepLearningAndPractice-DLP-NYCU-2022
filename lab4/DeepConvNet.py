import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepConvNet(nn.Module):
    def __init__(self, activation = nn.ELU()):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5)),
            nn.Conv2d(25, 25, kernel_size=(2, 1)),
            nn.BatchNorm2d(25),
            activation,
            nn.MaxPool2d((1,2)),
            nn.Dropout(0.5)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 2)),
            nn.BatchNorm2d(50),
            activation,
            nn.MaxPool2d((1,2)),
            nn.Dropout(0.5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 2)),
            nn.BatchNorm2d(100),
            activation,
            nn.MaxPool2d((1,2)),
            nn.Dropout(0.5)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 2)),
            nn.BatchNorm2d(200),
            activation,
            nn.MaxPool2d((1,2)),
            nn.Dropout(0.5)
        )

        self.flatten = nn.Flatten()
        self.FC = nn.Linear(9000, 2)
    
    def forward(self, x:torch.Tensor):
        out = x.reshape((x.shape[0], 1, 2, 750))
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.flatten(out)
        out = self.FC(out)
        return out