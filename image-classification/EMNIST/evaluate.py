from EMNIST.utils.dataloader import test_loader, train_loader
from EMNIST.models.emnist_net import EMNISTNet
from torch import nn
import torch

EMNISTModel = EMNISTNet()
EMNISTModel.load_state_dict(torch.load('../saved_model/emnist_model_state_dict.pth', map_location=torch.device('cpu')))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMNISTModel = EMNISTModel.to(device)

EMNISTModel.eval()
loss_fn = nn.CrossEntropyLoss()


def evaluate_model(data_loader):
    total = 0
    correct = 0
    total_loss = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = EMNISTModel(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total * 100
    return avg_loss, accuracy

# Evaluate on training data
train_loss, train_accuracy = evaluate_model(train_loader)
print(f'Average Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')

# Evaluate on test data
test_loss, test_accuracy = evaluate_model(test_loader)
print(f'Average Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')