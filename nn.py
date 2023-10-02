import torch
import torch.nn as tnn
import torch.nn.functional as F

class nn(tnn.Module):

    def __init__(self, k :int):
        super().__init__()

        self.fc1 = tnn.Linear(18+3*k, 120)  # 5*5 from image dimension
        self.fc2 = tnn.Linear(120, 64)
        self.fc3 = tnn.Linear(64, 15)

        self.relu = tnn.ReLU()

    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
       
        return self.fc3(x)