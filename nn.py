import torch
import torch.nn as tnn
import torch.nn.functional as F

class nn(tnn.Module):

    def __init__(self, k :int):
        super().__init__()

        self.fc1 = tnn.Linear(84+6*k, 512) #ALWAYS have size one more because add number of actions left
        self.fc2 = tnn.Linear(512, 256)
        self.fc3 = tnn.Linear(256, 80)
        self.fc4 = tnn.Linear(80, 17)

        self.relu = tnn.LeakyReLU()
        self.tanh = tnn.Tanh()

    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
       
        return self.fc4(x)