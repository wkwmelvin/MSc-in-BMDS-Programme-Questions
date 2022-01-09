import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18 as _resnet18
from resnet_no_bn import resnet18
import math

class resnet_feature_extractor(nn.Module):
    def __init__(self):
        super(resnet_feature_extractor, self).__init__()
        
        # self.model = torch.hub.load('pytorch/vision:v0.11.2', 'resnet18', pretrained = False)
        self.model = resnet18()

        num_ftrs = self.model.fc.in_features
        self.model.conv1 = nn.Conv2d(100, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(num_ftrs, 128)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.model(x)
        x = self.relu(x)
        # x = torch.reshape(x, (-1, 32, 32))
        
        return x


#note: the distribution pooling filter implmentation taken from the Author's github
class DistributionPoolingFilter(nn.Module):

    __constants__ = ['num_bins', 'sigma']

    def __init__(self, num_bins=21, sigma=0.5):
        super(DistributionPoolingFilter, self).__init__()
        
        self.num_bins = num_bins
        
        self.sigma = sigma
        self.alfa = 1/math.sqrt(2*math.pi*(sigma**2))
        self.beta = -1/(2*(sigma**2))
        
        sample_points = torch.linspace(0,1,steps=num_bins, dtype=torch.float32, requires_grad=False)
        
        self.register_buffer('sample_points', sample_points)


    def extra_repr(self):
    		return 'num_bins={}, sigma={}'.format(
    			self.num_bins, self.sigma
    		)


    def forward(self, data):
        
        batch_size = 1
        
        num_instances, num_features = data.size()
        
        sample_points = self.sample_points.repeat(batch_size, num_instances, num_features, 1)
        # sample_points.size() --> (batch_size,num_instances,num_features,num_bins)
        
        data = torch.reshape(data,(batch_size,num_instances,num_features,1))
        # data.size() --> (batch_size,num_instances,num_features,1)
        
        diff = sample_points - data.repeat(1,1,1,self.num_bins)
        diff_2 = diff**2
        # diff_2.size() --> (batch_size,num_instances,num_features,num_bins)
        
        result = self.alfa * torch.exp(self.beta*diff_2)
        # result.size() --> (batch_size,num_instances,num_features,num_bins)
        
        out_unnormalized = torch.sum(result,dim=1)
        # out_unnormalized.size() --> (batch_size,num_features,num_bins)
        
        norm_coeff = torch.sum(out_unnormalized, dim=2, keepdim=True)
        # norm_coeff.size() --> (batch_size,num_features,num_bins)
        
        out = out_unnormalized / norm_coeff
        # out.size() --> (batch_size,num_features,num_bins)
        
        return out


class transformation(nn.Module):
    def __init__(self):
        super(transformation, self).__init__()

        # self.layer1 = nn.Linear(32*21, 336)
        self.layer1 = nn.Linear(2688, 336)
        self.layer2 = nn.Linear(336, 168)
        self.fc = nn.Linear(168, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.fc(x)

        return x

class MIL_architecture(nn.Module):
    def __init__(self):
        super(MIL_architecture, self).__init__()

        self.feature_extr = resnet_feature_extractor()
        self.distr_pool = DistributionPoolingFilter()
        self.transformer = transformation()

    def forward(self, x):
        x = self.feature_extr(x)
        x = self.distr_pool(x)
        x = torch.flatten(x, 1)
        x = self.transformer(x)

        return x



