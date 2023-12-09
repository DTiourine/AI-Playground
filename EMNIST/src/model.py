import torch.nn as nn
import torch.nn.functional as F
import torch

class EMNISTNet(nn.Module):
    def __init__(self):
        super(EMNISTNet, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 62),
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

EMNISTModel = EMNISTNet()

#dummy_input = torch.zeros((64, 1, 28, 28), dtype=torch.float32)
#dummy_output = EMNISTModel(dummy_input)