import torch
import torch.nn as tnn
import torch.nn.functional as F

class nn(tnn.Module):

    def __init__(self, k :int):
        super().__init__()

        self.fc1 = tnn.Linear(143, 512)
        self.norm1 = tnn.BatchNorm1d(512)
        self.fc2 = tnn.Linear(512, 256)
        self.norm2 = tnn.BatchNorm1d(256)
        self.fc3 = tnn.Linear(256, 80)
        self.norm3 = tnn.BatchNorm1d(80)
        self.fc4 = tnn.Linear(80, 16)

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