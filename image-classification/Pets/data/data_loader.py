from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_set = datasets.OxfordIIITPet(
    root='./data',
    download=True,
    split='trainval',
    transform=transform
)

test_set = datasets.OxfordIIITPet(
    root='./data',
    download=True,
    split='test',
    transform=transform
)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)