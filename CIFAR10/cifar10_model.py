from torch import nn
from cifar10_data_loader import train_set

class CIFAR10Net(nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels = 9, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.linear = nn.Linear(in_features=15*15*9, out_features= 10)

    def forward(self, x):
        print(f'Original shape: {x.shape}')
        x = self.conv1(x)
        print(f'After conv: {x.shape}')
        x = self.pool1(x)
        print(f'After pool: {x.shape}')
        return x

Cifar10Model = CIFAR10Net()
x, y = train_set[0]
print(Cifar10Model(x))