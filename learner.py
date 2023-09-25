import torch
import torch.nn as tnn
import torch.nn.functional as F
import numpy as np

from nn import nn

loss = tnn.MSELoss()

class Learner():

    def __init__(self, k: int, gamma: float):

        self.prediction_nn = nn(k)
        self.target_nn = nn(k)

        self.y_hat=None
        self.y=None

        self.action = None
        self.action2 = None
        self.gamma = gamma

    def predict(self, x: torch.tensor, epsilon: float):

        out = self.prediction_nn.forward(x)

        #POLICY
        proba=np.random.uniform(0,1)


        if proba>epsilon:
            self.action=out.argmax()
        else:
            self.action=int(np.random.randint(0,15))

        self.y_hat=out[:,self.action][0]


    def target(self, x: torch.tensor, r: int):

        get_a=self.target_nn.forward(x)
        self.action2=get_a.max(dim=1).indices

        out2=self.prediction_nn.forward(x)
        self.out2=out2

        Q_next=out2[:,self.action2][0][0]

        #y:     
        print(r)
        print(Q_next)
        self.y=r+ self.gamma*Q_next


    def get_loss(self):

        return loss(self.y, self.y_hat)

