from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()

train_set = datasets.CIFAR10(
    root='./data',
    download=True,
    train=True,
    transform=transform
)

test_set = datasets.CIFAR10(
    root='./data',
    download=True,
    train=False,
    transform=transform
)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)