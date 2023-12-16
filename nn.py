import torch
import torch.nn as tnn
import torch.nn.functional as F

class nn(tnn.Module):

    def __init__(self, k :int):
        super().__init__()

        self.fc1 = tnn.Linear(38+2*k+1, 240) #ALWAYS have size one more because add number of actions left
        self.fc2 = tnn.Linear(240, 120)
        self.fc3 = tnn.Linear(120, 25)

        self.relu = tnn.LeakyReLU()

    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
       
        return self.fc3(x)