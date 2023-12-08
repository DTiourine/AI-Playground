from torch import nn

class PetNet(nn.Module):
    def __init__(self):
        self.Conv2d = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=3, stride=1)
        self.MaxPool = nn.MaxPool2d(kernel_size=3)
        self.Flatten()
        self.Linear()
