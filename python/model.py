import torch
from torch import nn
import torch.nn.functional as F
import torchvision as tv
from torchvision.transforms import v2
from diffusers import utils
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm
from copy import deepcopy


class mlp(nn.Module):
    def __init__(self,PARAMS):
        """
        A simple mlp model to predict time series
        modeled as regression problem
        param int input_size: size of input 
        return torch.tensor prediction
        """
        super().__init__()
        self.fc1      = nn.Linear(PARAMS.input_size,128)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2      = nn.Linear(128,64)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3      = nn.Linear(64,1)

    def forward(self,x):
        # convolutional layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        # output
        x = self.fc3(x)
        return x

class lstm(nn.Module):
    def __init__(self,PARAMS):
        """
        A simple lstm model to predict time series
        param int hidden size: the size of hidden layer
        param int lstm_input_size: size of input 
        return torch.tensor prediction 
        """
        super().__init__()
        self.lstm      = nn.LSTM(input_size=PARAMS.input_size, hidden_size=PARAMS.hidden_size, num_layers=1, batch_first=True) # ,dropout=0.25
        # self.batchnorm = nn.BatchNorm1d(hidden_size)
        self.linear    = nn.Linear(PARAMS.hidden_size, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

class cnn_lstm(nn.Module):
    def __init__(self,PARAMS):
        """
        A simple cnn_lstm model to predict time series
        param int hidden size: the size of lstm hidden layer
        param int lstm_input_size: size of input 
        param int channels_c1: number of filters for 1st convolution
        param int channels_c2: number of filters for 2nd convolution
        return torch.tensor prediction 
        
        """
        super().__init__()
        
        # convolution
        self.conv1    = nn.Conv1d(in_channels=1, out_channels=PARAMS.conv_c1,kernel_size=4, padding=2, padding_mode='zeros')
        self.conv2    = nn.Conv1d(in_channels=PARAMS.conv_c1, out_channels=PARAMS.conv_c2,kernel_size=4, padding=2, padding_mode='zeros')
        self.maxpool  = nn.MaxPool1d(kernel_size=2, stride=1) # kernel size =3 for conv2
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.fc1      = nn.Linear(PARAMS.conv_c1 * PARAMS.input_size, 512)
        self.fc2      = nn.Linear(512, 128 )
        # lstm
        self.lstm     = nn.LSTM(input_size=128, hidden_size=PARAMS.hidden_size, num_layers=1, batch_first=True) # ,dropout=0.25
        self.linear   = nn.Linear(PARAMS.hidden_size, 1)
        
    def forward(self, x):

        # Convolution
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool(x) # tmp
        
        # x = self.conv2(x)
        # x = F.relu(x)
        # x = self.maxpool(x)
        x = self.dropout1(x)
        x = torch.flatten(x,-2) # concatenate feature maps
        
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        x = torch.stack([x],dim = 0) # convert 1d tensor to 2d for lstm input

        x    = torch.permute(x, (1, 0, 2)) # arrange dimensions so that matches lstm input for batch processing
        x, _ = self.lstm(x)
        x    = self.linear(x)
        return x