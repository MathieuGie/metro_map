import torch
import torch.nn as tnn
import torch.nn.functional as F
import numpy as np

from nn import nn

class Learner():

    def __init__(self, k: int, gamma: float):

        self.prediction_nn = nn(k)
        self.target_nn = nn(k)

        self.y_hat=None
        self.y=None

        self.action = None
        self.action2 = None
        self.gamma = gamma

        self.loss = tnn.MSELoss()

    def predict(self, x: torch.tensor, epsilon: float):

        with torch.no_grad():
            out = self.prediction_nn.forward(x)

            #POLICY
            proba=np.random.uniform(0,1)

            if proba>epsilon:
                self.action=int(out.argmax())
            else:
                self.action=int(np.random.randint(0,17))

            #print("action", self.action)

            self.y_hat=out[:,self.action][0]

            #print("y_hat", self.y)

    def predict_for_replay(self, x: torch.tensor, action: int):

        out = self.prediction_nn.forward(x)
        self.y_hat=out[:,action][0]


    def target(self, x: torch.tensor, r: int, final):

        with torch.no_grad():

            get_a=self.prediction_nn.forward(x)

            self.action2=get_a.max(dim=1).indices


            #print("action2", self.action2)

            out2=self.target_nn.forward(x)
            self.out2=out2

            Q_next=out2[:,self.action2][0][0]

            #y
            if final is True:
                self.y = torch.tensor(r, dtype=torch.float32)
            else:
                self.y= torch.tensor(r, dtype=torch.float32) + self.gamma*Q_next

        #print("y", self.y_hat)


    def get_loss(self):

        return self.loss(self.y, self.y_hat)

