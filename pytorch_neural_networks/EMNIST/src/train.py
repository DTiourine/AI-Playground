import torch
from torch import nn
from EMNIST.src.model import EMNISTNet
from EMNIST.src.data.data_loader import train_loader

#print(torch.cuda.is_available())
EMNIST_Model = EMNISTNet()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMNIST_Model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=EMNIST_Model.parameters(), lr=0.1)

EPOCHS = 100

for epoch in range(EPOCHS):

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        #Forward pass
        outputs = EMNIST_Model(inputs)
        loss = loss_fn(outputs, labels)

        #Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{EPOCHS}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

torch.save(EMNIST_Model.state_dict(), '../saved_model/emnist_model_state_dict.pth')
