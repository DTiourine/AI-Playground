# Neural Network Architecture

import torch.nn as nn
class FashionCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=3),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=200, out_features=120),
            nn.Linear(in_features=120, out_features=10),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.layers(x)
        return x