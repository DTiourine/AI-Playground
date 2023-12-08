import torch
from emnist_model import EMNISTNet

EMNIST_Model = EMNISTNet()
EMNIST_Model.load_state_dict(torch.save(torch.load('emnist_model_state_dict.pth')))
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total

def calculate_loss(loader, model, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_accuracy = 0.0
    total_samples = 0

    with torch.no_grad():  # Inference mode, no gradients needed
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_accuracy += calculate_accuracy(outputs, labels) * inputs.size(0)
            total_samples += inputs.size(0)

    avg_loss = total_loss / len(loader)
    avg_accuracy = total_accuracy / total_samples
    return avg_loss, avg_accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

test_loss, test_accuracy = calculate_loss(test_loader, model, criterion, device)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')