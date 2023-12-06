import torch
from torch import nn
from emnist_model import EMNISTNet
from emnist_data_loader import train_loader


EMNIST_Model = EMNISTNet()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=EMNIST_Model.parameters(), lr=0.1)

EPOCHS = 100

for epoch in range(EPOCHS):

    running_loss = 0.0
    last_loss = 0.0

    for batch in train_loader:
        pass
