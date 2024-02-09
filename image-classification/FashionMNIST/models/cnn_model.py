import torch.nn as nn
import torch.nn.functional as F

# Define the CNN model
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        # Convolutional layer (sees 28x28x1 image tensor)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # Convolutional layer (sees 14x14x16 tensor)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Linear layer (32 * 7 * 7 = 1568 input features, 128 output features)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        # Linear layer (128 -> 10 classes)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten image input
        x = x.view(-1, 32 * 7 * 7)
        # Add dropout layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x