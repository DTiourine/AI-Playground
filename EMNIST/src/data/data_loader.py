import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Get the directory in which data_loader.py is located
base_path = os.path.dirname(__file__)

# Define the path for the dataset relative to the location of data_loader.py
dataset_path = os.path.join(base_path, 'data/data')

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Download and load the training data
train_set = datasets.EMNIST(
    root='./data',
    split='byclass',
    download=True,
    train=True,
    transform=transform)

test_set = datasets.EMNIST(
    root = './data',
    split='byclass',
    download=True,
    train=False,
    transform=transform
)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)