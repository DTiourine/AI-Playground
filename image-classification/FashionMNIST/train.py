import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn_model import FashionCNN
from utils.dataloader import load_datasets
import numpy as np
from sklearn.metrics import f1_score


def compute_metrics(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    accuracy = np.mean(preds == labels)
    f1 = f1_score(labels, preds, average='macro')
    return accuracy, f1

def train_model(model, training_loader, criterion, optimizer, epochs=50):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        all_preds = []
        all_labels = []
        for images, labels in training_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        accuracy, f1 = compute_metrics(all_preds, all_labels)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(training_loader)}, Accuracy: {accuracy*100:.2f}%, F1 Score: {f1:.2f}')

def main():
    training_loader, test_loader = load_datasets('data')

    model = FashionCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, training_loader, criterion, optimizer, epochs=10)

    torch.save(model.state_dict(), 'trained_models/fashion_cnn.pth')

if __name__ == "__main__":
    main()


"""
#Instantiate model, define loss, define optimizer
import torch
from torch import nn
from FashionMNIST.src.model import FMNISTModel
from FashionMNIST.data.data_loader import train_dataloader

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
"""