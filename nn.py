import torch
import torch.nn as tnn
import torch.nn.functional as F

class nn(tnn.Module):

    def __init__(self, k :int):
        super().__init__()

        self.fc1 = tnn.Linear(125, 256)
        self.norm1 = tnn.BatchNorm1d(256)
        self.fc2 = tnn.Linear(256, 128)
        self.norm2 = tnn.BatchNorm1d(128)
        self.fc3 = tnn.Linear(128, 32)
        self.norm3 = tnn.BatchNorm1d(32)
        self.fc4 = tnn.Linear(32, 8)

        self.relu = tnn.LeakyReLU()
        self.tanh = tnn.Tanh()

    def forward(self, x):

        if x.shape[0]==1:

            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.relu(self.fc3(x))

        else:

            x = self.relu(self.norm1(self.fc1(x)))
            x = self.relu(self.norm2(self.fc2(x)))
            x = self.relu(self.norm3(self.fc3(x)))
       
        return self.fc4(x)