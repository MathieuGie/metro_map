import torch
import torch.nn as nn
import torch.nn.functional as F

class nn(nn.Module):

    def __init__(self, k :int):
        super(nn, self).__init__()

        self.fc1 = nn.Linear(12+4*k, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 15)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
       
        return self.fc3(x)