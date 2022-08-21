import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Transformer
from torch import nn
from torch.utils.data import DataLoader
import math

# models for experiments 
    
class M_convED_10(torch.nn.Module):
    def __init__(self, input_channels = 5, length = 160, output_dim = 40, conv_channels = 64):
        super(M_convED_10, self).__init__()
        
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.length = length
        self.channels = conv_channels
        
        self.conv = nn.Sequential(
            torch.nn.Conv1d(in_channels=self.input_channels, out_channels=self.channels, kernel_size=3, padding=1),
            torch.nn.Conv1d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1),
            torch.nn.AvgPool1d(2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1),
            torch.nn.Conv1d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1),
            torch.nn.AvgPool1d(2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1),
            torch.nn.Conv1d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1),
            torch.nn.AvgPool1d(2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1),
            torch.nn.Conv1d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1),
            torch.nn.AvgPool1d(2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1),
            torch.nn.Conv1d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1),
            torch.nn.AvgPool1d(2),
            torch.nn.ReLU(),
        )
        self.fc1 = Linear(self.length//(2**5)*self.channels,  output_dim*4)
        self.fc2 = Linear(output_dim*4, output_dim)
        self.final_bn = torch.nn.BatchNorm1d(output_dim, momentum=0.01)
                
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.final_bn(x)
        return x
    
class M_convED_5(torch.nn.Module):
    def __init__(self, input_channels = 5, length = 160, output_dim = 40, conv_channels = 64):
        super(M_convED_5, self).__init__()
        
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.length = length
        self.channels = conv_channels
        
        self.conv = nn.Sequential(
            torch.nn.Conv1d(in_channels=self.input_channels, out_channels=self.channels, kernel_size=3, padding=1),
            torch.nn.AvgPool1d(2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1),
            torch.nn.AvgPool1d(2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1),
            torch.nn.AvgPool1d(2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1),
            torch.nn.AvgPool1d(2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1),
            torch.nn.AvgPool1d(2),
            torch.nn.ReLU(),
        )
        self.fc1 = Linear(self.length//(2**5)*self.channels,  output_dim*4)
        self.fc2 = Linear(output_dim*4, output_dim)
        self.final_bn = torch.nn.BatchNorm1d(output_dim, momentum=0.01)
                
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.final_bn(x)
        return x
    
class M_GRU(nn.Module):
    def __init__(self, length = 160, output_dim = 40, n_hidden = 64):
        super(M_GRU, self).__init__()
        
        self.length = length
        self.output_dim = output_dim
        self.n_hidden = n_hidden
        self.gru_cell = nn.GRU(input_size=5, hidden_size=n_hidden, 
                               num_layers=2, dropout=0.5, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(self.length*self.n_hidden*2,  output_dim*4)
        self.fc2 = nn.Linear(output_dim*4, output_dim)
        self.final_bn = torch.nn.BatchNorm1d(output_dim, momentum=0.01)

    def forward(self, x):
        x = torch.transpose(x,-1,-2)
        x, _ = self.gru_cell(x)
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.final_bn(x)
        return x

class M_RNN(nn.Module):
    def __init__(self, length = 160, output_dim = 40, n_hidden = 64):
        super(M_RNN, self).__init__()
        
        self.length = length
        self.output_dim = output_dim
        self.n_hidden = n_hidden
        
        self.rnn_cell = nn.RNN(input_size=5, hidden_size=n_hidden, 
                               num_layers=2, dropout=0.5, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(self.length*self.n_hidden*2,  output_dim*4)
        self.fc2 = nn.Linear(output_dim*4, output_dim)
        self.final_bn = torch.nn.BatchNorm1d(output_dim, momentum=0.01)

    def forward(self, x):
        x = torch.transpose(x,-1,-2)
        x, _ = self.rnn_cell(x)
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.final_bn(x)
        return x
    