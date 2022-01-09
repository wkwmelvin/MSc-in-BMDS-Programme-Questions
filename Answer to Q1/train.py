import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18 as _resnet18
from dataset import MNIST_BAG, MNIST_BAG_VAL
from model import MIL_architecture
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MIL_architecture().to(device)

#hyperparameters
lr = 1e-4
batch_size = 1
weight_decay = 0.0005
criterion = nn.L1Loss()
num_epochs = 100

# optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

total_train_loss = []
total_val_loss = []

dataset = MNIST_BAG()
valset = MNIST_BAG_VAL()
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)
dataloader_val = DataLoader(dataset=valset, batch_size=batch_size, shuffle=True, drop_last=True)

epoch_ = []

for epoch in range(num_epochs):
    train_loss = 0
    val_loss = 0
    
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        #forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        current_loss_train = loss.item()
        train_loss += current_loss_train
        
    train_loss /= 500
    
    for i, (inputs, labels) in enumerate(dataloader_val):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        #forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        current_loss_val = loss.item()
        val_loss += current_loss_val
        
    val_loss /= 100
    
    total_train_loss.append(train_loss)
    total_val_loss.append(val_loss)
    
    epoch_.append(epoch+1)
    
    # print(f"Epochs: {epoch+1}/{num_epochs}, Loss: {train_loss}, Output: {outputs}")
    print(f"Epochs: {epoch+1}/{num_epochs}, Loss_train: {train_loss}, loss_val: {val_loss}")

plt.plot(epoch_[0:51], total_train_loss[0:51], label = 'Training Loss')
plt.plot(epoch_[0:51], total_val_loss[0:51], label = 'Validation Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')