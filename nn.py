import torch
import torch.nn as tnn
import torch.nn.functional as F

class nn(tnn.Module):

    def __init__(self, k :int):
        super().__init__()

        self.fc1 = tnn.Linear(90, 256) #ALWAYS have size one more because add number of actions left
        self.fc2 = tnn.Linear(256, 128)
        self.fc3 = tnn.Linear(128, 80)
        self.fc4 = tnn.Linear(80, 17)

        self.relu = tnn.LeakyReLU()
        self.tanh = tnn.Tanh()

    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
       
        return self.fc4(x)