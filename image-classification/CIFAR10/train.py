from utils.dataloader import load_datasets
import torch
import torch.nn as nn
import torch.optim as optim
from models.cifar10_net import CIFAR10Net
import numpy as np
from sklearn.metrics import f1_score

def compute_metrics(pred, labels):
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
    print(f'Cuda available: {torch.cuda.is_available()}')
    training_loader, test_loader = load_datasets('data')

    model =CIFAR10Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, training_loader, criterion, optimizer, epochs=50)

    torch.save(model.state_dict(), 'trained_models/cifar10_net.pth')
