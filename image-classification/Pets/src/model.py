from torch import nn
import torch

class PetNet(nn.Module):
    def __init__(self):
        super(PetNet, self).__init__()
        self.Conv2d = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=3, stride=1)
        self.MaxPool2d = nn.MaxPool2d(kernel_size=3)
        self.Flatten() = nn.Flatten()
        self.Linear() = nn.Linear(in_features=9*74*74, out_features=)

    def forward(self, x):
        print(f'Shape at input layer: {x.shape}')
        x = self.Conv2d(x)
        print(f'Shape after Conv2d: {x.shape}')
        x = self.MaxPool2d(x)
        print(f'Shape after Maxpool2d: {x.shape}')
        x = self.Flatten(x)
        print(f'Shape after Flatten: {x.shape}')
        x = self.Linear(x)
        print(f'Shape after Linear: {x.shape}')

dummy_input = torch.randn(1, 3, 224, 224)
PetModel = PetNet()
PetModel.eval()
dummy_output = PetModel(dummy_input)