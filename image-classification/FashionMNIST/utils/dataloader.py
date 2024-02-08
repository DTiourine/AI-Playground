# utils/dataloader.py
from torchvision import datasets, transforms
import torch

def load_datasets(data_dir='data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download and load the training data
    training_set = datasets.FashionMNIST(data_dir, download=True, train=True, transform=transform)
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=64, shuffle=True)

    # Download and load the test data
    test_set = datasets.FashionMNIST(data_dir, download=True, train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

    return training_loader, test_loader
