import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18 as _resnet18
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MNIST_BAG(Dataset):
    def __init__(self):
        self.data = np.load('data_train.npy', allow_pickle=True)
        self.data = self.data.item()
        
        self.labels = np.load('labels_train.npy', allow_pickle=True)
        self.labels = self.labels.item()
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        bag, label = self.data[idx], self.labels[idx]
        
        return bag, label