from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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