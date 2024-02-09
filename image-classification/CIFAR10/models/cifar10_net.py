from torch import nn
import torch
class CIFAR10Net(nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels = 9, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(in_features=15*15*9, out_features= 10)

    def forward(self, x):
        #print(f'Original shape: {x.shape}')
        x = self.conv1(x)
        #print(f'After conv: {x.shape}')
        x = self.pool1(x)
        #print(f'After pool: {x.shape}')
        x = self.flatten(x)
        #print(f'After flatten: {x.shape}')
        x = self.linear(x)
        #print(f'After linear: {x.shape}')
        return x

Cifar10Model = CIFAR10Net()
dummy_input = torch.randn(64, 3, 32, 32)
dummy_output = Cifar10Model(dummy_input)

print(dummy_output.shape)