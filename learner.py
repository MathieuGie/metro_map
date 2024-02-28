import torch
import torch.nn as tnn
import torch.nn.functional as F
import numpy as np

from nn import nn
from env import Environment

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

    def predict(self, x: torch.tensor, epsilon: float, loc, env: Environment):

        with torch.no_grad():
            out = self.prediction_nn.forward(x)

            #POLICY
            proba=np.random.uniform(0,1)

            if proba>epsilon:
                self.action=int(out.argmax())

            elif proba>epsilon/2: #Fully greedy action taking (to direct the learning towards better actions)

                if proba>3*epsilon/4:
                    constant = 1
                else:
                    constant = 0.4

                neighbours = [(1,0), (0,1), (-1,0), (0, -1), (np.sqrt(2)/2, np.sqrt(2)/2), (-np.sqrt(2)/2, -np.sqrt(2)/2), (np.sqrt(2)/2, -np.sqrt(2)/2), (-np.sqrt(2)/2, np.sqrt(2)/2)]
                best_action = 0
                best=0
                for possible in range(8):

                    new_loc = (loc[0] + int(8*neighbours[possible][0]), loc[1] + int(8*neighbours[possible][1]))
                    if new_loc[0]<env.size/2 and new_loc[0]>-env.size/2 and new_loc[1]<env.size/2 and new_loc[1]>-env.size/2:
                        dens, _ = env.get_dense_around(new_loc, coef=constant)
                        dens_occupied  = env.get_share_already_served(new_loc, coef=constant)

                        if dens-dens_occupied>best:
                            best = dens-dens_occupied
                            best_action = possible

                self.action = best_action

            else:
                self.action=int(np.random.randint(0,8))

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

