from env import ReplayBuffer, Environment
from learner import Learner

import numpy as np
import torch 
from collections import deque
import random
import matplotlib.pyplot as plt

class Coordinator:

    def __init__(self, size, n_simulations, city_center, city_facts, first_station, metro_facts, buffer_size, learning_var, n_iter):

        self.environment = Environment(size, n_simulations, city_center, city_facts, first_station, metro_facts)
        self.buffer = ReplayBuffer(buffer_size)

        self.epsilon = learning_var["epsilon"]
        self.epsilon_decay = learning_var["epsilon_decay"]
        self.tau = learning_var["tau"]
        self.gamma = learning_var["gamma"]
        self.update_target_interval = learning_var["update_target_interval"]

        self.learner = Learner(self.environment.max_connected, self.gamma)


        self.optimiser =torch.optim.Adam(self.learner.prediction_nn.parameters(), lr=0.001)
        self.average_reward = 0
        self.total_reward = 0
        self.time = 0

        self.n_iter = n_iter

    
    def feed_play(self, n_selected:int, n_allowed_per_play: int):

        L = None
        actions_left=1 #this will decrease the more actions we deecide to do.

        #Select the desired station
        done=set()
        done.add(-1) #Add a starting fake station

        for _ in range(n_selected):

            #Make state s: (needed at every new step because changes are made)
            station = self.environment.select_station()
            state = self.environment.make_state(station)

            #Play action:
            remaining=torch.zeros(1,1)
            remaining[0,0]=actions_left

            row = state[0,:]
            row = row.reshape(1,-1)

            vec = torch.cat((remaining, row), axis=1)

            self.learner.predict(vec, self.epsilon)
            action = self.learner.action


            #When not allowed to play the action
            if action != 0 and actions_left<=0:
                actions_left=0
                action = 0
                r = 0

            #When action is playable
            else:
                if action != 0:
                    actions_left-=1/n_allowed_per_play

                self.environment.change_metro(station, action)
                r = self.environment.get_reward()

                self.environment.change_metropolis()


            STATE = vec
            ACTION = action
            REWARD = r
            self.average_reward+=r
            self.total_reward+=1

            ### REMAINING ###
            remaining=torch.zeros(1,1)
            remaining[0,0]=actions_left

            #Make state s':
            new_state = self.environment.make_state(station)

            new_row = new_state[0,:]
            new_row = new_row.reshape(1,-1)

            vec = torch.cat((remaining, new_row), axis=1)

            NSTATE = vec
            self.buffer.push((STATE, ACTION, REWARD, NSTATE))

            self.learner.target(vec, r)


            if L is None:
                L = self.learner.get_loss()

            else:
                L += self.learner.get_loss()
        
        ###Now sample:
        samples = self.buffer.sample(64)

        for sample in samples:

            self.learner.predict_for_replay(sample[0], sample[1])
            self.learner.target(sample[3], sample[2])
            L+=self.learner.get_loss()

        return L/(n_selected+64) #Need to divide
    
    def backprop(self, L, time):

        #Backprop on nn
        self.optimiser.zero_grad()
        L.backward()
        self.optimiser.step()

        #Update nn_target

        if time%self.update_target_interval==0:
            print("UPDATING TARGET")
            for target_param, param in zip(self.learner.target_nn.parameters(), self.learner.prediction_nn.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def step(self, time_target:int):

        for i in range(self.n_iter):

            print("iteration_coord:", i)

            self.time+=1
            L = self.feed_play(10, 7)
            self.epsilon*=self.epsilon_decay

            self.backprop(L, self.n_iter*time_target+self.time)

    def reset(self):

        self.average_reward /= self.total_reward
        out = self.average_reward

        self.average_reward=0
        self.time=0

        self.environment.reset()

        return out
    
    def display(self):

        frame = self.environment.metropolis.display()
        plt.imshow(frame, cmap='viridis', origin='lower')

        LINES = self.environment.metro.display(True)
        print(LINES)

        # Predefined list of colors.
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        for idx, (key, line) in enumerate(LINES.items()):
            color = colors[idx % len(colors)]  # Cycle through colors if there are more lines than colors.
            for i in range(len(line)-1):
                x_values = [line[i][0], line[i+1][0]]
                y_values = [line[i][1], line[i+1][1]]
                plt.plot(x_values, y_values, color, label=f"Line {key}" if i == 0 else "", marker="o")

        plt.colorbar(label='Density')
        plt.title('Density and Points')

        plt.legend()
        plt.show()
    

###############################


metro_params={

    #Distances
    "speed_metro" : 8,
    "speed_change" : 2,
    "speed_walk" : 1,

    #Times
    "waiting_for_train": 5,
    "waiting_when_stopping": 1,

    "max_connected" : 2, # A change station has at most 2 connections

    "r_walking" : 60,
    "k_walking" : 2,

    "p_selecting_station":0.95 # chance of prolongating a line instead of randomly selecting a station
}

city_params={
    "p_center" : 0.3,
    "p_new" : 0.2,
    "p_station": 0.2,
    "at_most_new":3
}

learning_var={
    "epsilon":0.95,
    "epsilon_decay":0.999,
    "tau":0.6,
    "update_target_interval":20,
    "gamma":0.98

}

coord = Coordinator(200, 1000, (0,0), city_params, (0,0), metro_params, 1000, learning_var, 6)


all = []
time_target=0

for i in range(1000):

    print("reset", i)
    coord.step(time_target)
    time_target+=1 #Keep track for the update of the target

    r = coord.reset()
    print("r:", r)
    print("epsilon:", coord.epsilon)
    #print(coord.stations_network.all_stations)
    all.append(r)

    # Generate your plot with the current state of your_list
    plt.figure()
    plt.plot(all)
    plt.title('Rewards')
    
    # Save the figure to the same file location every time
    plt.savefig('reward.png')
    
    # Close the figure to free memory
    plt.close()

#Last one to plot:
coord.step(time_target)


print("all_rewards:", all)

coord.display()