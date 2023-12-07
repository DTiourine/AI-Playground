from CIFAR10.src.data.cifar10_data_loader import train_loader
from CIFAR10.src.cifar10_model import CIFAR10Net
from torch import nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR10Model = CIFAR10Net().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(CIFAR10Model.parameters(), lr=0.05)

EPOCHS = 10
for epoch in range(EPOCHS):
    for i, batch in enumerate(train_loader):
        CIFAR10Model.train()

        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = CIFAR10Model(inputs)

        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{EPOCHS}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

torch.save(CIFAR10Model.state_dict(), '../saved_model/CIFAR10Model_state_dict.pth')