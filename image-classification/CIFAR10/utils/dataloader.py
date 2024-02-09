from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_datasets(data_dir="data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    training_set = datasets.CIFAR10(root=data_dir, download=True, train=True, transform=transform)
    training_loader = DataLoader(training_set, batch_size=64, shuffle=True)

    test_set = datasets.CIFAR10(root=data_dir, download=True, train=False, transform=transform)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    return training_loader, test_loader