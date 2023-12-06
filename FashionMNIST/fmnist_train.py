#Instantiate model, define loss, define optimizer
import torch
from torch import nn
from fmnist_model import FMNISTModel
from fmnist_data_loader import train_dataloader

import torch.optim as optim

model = FMNISTModel()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop
epochs = 400

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for batch_data, batch_labels in train_dataloader:
        outputs = model(batch_data)
        loss = loss_fn(outputs, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        _, predicted_labels = torch.max(outputs, dim=1)  # Get predicted labels
        correct_predictions += (predicted_labels == batch_labels).sum().item()
        total_samples += len(batch_labels)

    average_loss = total_loss / len(train_dataloader)
    accuracy = correct_predictions / total_samples  # Calculate accuracy
    print(f"Epoch [{epoch + 1}/{epochs}] - Average Loss: {average_loss:.4f} - Accuracy: {accuracy:.4f}")
