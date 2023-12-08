from CIFAR10.src.data.cifar10_data_loader import test_loader
from cifar10_model import CIFAR10Net
from torch import nn
import torch

CIFAR10Model = CIFAR10Net()
CIFAR10Model.load_state_dict(torch.load('../saved_model/CIFAR10Model_state_dict.pth'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CIFAR10Model = CIFAR10Model.to(device)

CIFAR10Model.eval()
loss_fn = nn.CrossEntropyLoss()

total = 0
correct = 0
total_loss = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = CIFAR10Model(inputs)

        loss = loss_fn(outputs, labels)
        total_loss += loss

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate average loss and accuracy
avg_loss = total_loss / len(test_loader)
accuracy = correct / total * 100

print(f'Average Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
