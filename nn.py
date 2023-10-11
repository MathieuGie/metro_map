import torch
import torch.nn as tnn
import torch.nn.functional as F

class nn(tnn.Module):

    def __init__(self, k :int):
        super().__init__()

        self.fc1 = tnn.Linear(34+3*k, 240) #ALWAYS have size one more because add number of actions left
        self.fc2 = tnn.Linear(240, 64)
        self.fc3 = tnn.Linear(64, 25)

        self.relu = tnn.ReLU()

    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
       
        return self.fc3(x)