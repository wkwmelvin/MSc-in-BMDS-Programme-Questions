import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18 as _resnet18
from dataset import MNIST_BAG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ()

lr = 0.0001
batch_size = 1
weight_decay = 0.0005
criterion = nn.L1Loss()

#optimizer = optim.Adam(model.parameters, lr=lr, weight_decay=weight_decay)

dataset = MNIST_BAG()
#train_features, train_labels = 