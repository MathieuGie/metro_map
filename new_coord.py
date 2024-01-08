from env import ReplayBuffer, Environment
from learner import Learner

import numpy as np
import torch 
from collections import deque
import random
import matplotlib.pyplot as plt
import time

class Coordinator:

    def __init__(self, size, n_simulations, city_center, city_facts, first_station, metro_facts, buffer_size, learning_var, n_iter, allowed_per_play, total_suggerable):

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

        self.allowed_per_play = allowed_per_play
        self.total_suggerable = total_suggerable

    def feed_play(self, n_selected:int, n_allowed_per_play: int):
        
        L = None

        for _ in range(2):

            if np.random.uniform(0,1)>self.epsilon:
                all_Q = {}
                with torch.no_grad():
                    for station in self.environment.metro.all_stations:

                        state = self.environment.make_state(station)
                        Q_values = self.learner.prediction_nn.forward(state)
                        all_Q[station] = Q_values

                averages = [(key, torch.max(value).item()) for key, value in all_Q.items()]
                best_station = max(averages, key=lambda x: x[1])[0]
            
            else:
                best_station = self.environment.select_station()

            state = self.environment.make_state(best_station)
            self.learner.predict(state, self.epsilon)
            action = self.learner.action

            #print("action", action)

            self.environment.change_metro(best_station, action)
            r = self.environment.get_reward(action)
            print(action, r)
            #self.environment.change_metropolis()

            new_state = self.environment.make_state(best_station)

            if len(self.environment.metro.all_stations)>=10:
                final=True
            else:
                final=False

            self.buffer.push((state, action, r, new_state, final))

            self.learner.target(new_state, r, final)

            self.average_reward+=r
            self.total_reward+=1

        #print("before", self.learner.y_hat)            
        
        ###Now sample:
        samples = self.buffer.sample(128)

        #print("SAMPLES", len(samples))

        for sample in samples:

            self.learner.predict_for_replay(sample[0], sample[1])
            self.learner.target(sample[3], sample[2], sample[4])
            #print(self.learner.y_hat)

            if L is None:
                L = self.learner.get_loss()

            else:
                L += self.learner.get_loss()

        return L/(len(samples)) #Need to divide

    
    def feed_play2(self, n_selected:int, n_allowed_per_play: int):
        
        L = None
        actions_left=1 #this will decrease the more actions we deecide to do.

        #Select the desired station
        done=set()
        done.add(-1) #Add a starting fake station

        for i in range(n_selected):

            print("ACTIIONS LEFT:", i, actions_left)

            #Make state s: (needed at every new step because changes are made)
            station = self.environment.select_station()
            state = self.environment.make_state(station)

            #Play action:
            remaining=torch.zeros(1,1)
            remaining[0,0]=actions_left

            row = state[0,:]
            row = row.reshape(1,-1)

            vec = torch.cat((remaining, row), axis=1)

            #print("state1", row.shape)

            self.learner.predict(vec, self.epsilon)
            action = self.learner.action


            #When not allowed to play the action
            if action != 0 and actions_left==0:
                actions_left=0
                action = 0
                r = -0.1

            elif action == 0 and actions_left==0:
                r = self.environment.get_reward()

            #When action is playable
            else:
                if action != 0:
                    actions_left-=1/n_allowed_per_play

                    if actions_left<0.0001:
                        actions_left=0

                self.environment.change_metro(station, action)
                r = self.environment.get_reward()

                self.environment.change_metropolis()

            #print("reward", r)
            STATE = vec
            ACTION = action
            REWARD = r
            ACTIONS_LEFT_TARGET = actions_left
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

            #print("state2", row)

            NSTATE = vec
            self.buffer.push((STATE, ACTION, REWARD, NSTATE, ACTIONS_LEFT_TARGET))

            self.learner.target(vec, r, actions_left)

            #print("Q_pred", self.learner.y_hat, "Q_target", self.learner.y)


            if L is None:
                L = self.learner.get_loss()

            else:
                L += self.learner.get_loss()
        
        ###Now sample:
        samples = self.buffer.sample(64)

        for sample in samples:

            self.learner.predict_for_replay(sample[0], sample[1])
            self.learner.target(sample[3], sample[2], sample[4])
            L+=self.learner.get_loss()

        return L/(n_selected+64) #Need to divide
    
    def backprop(self, L, t):

        #Backprop on nn
        #print(L)
        #time.sleep(2)
        self.optimiser.zero_grad()
        L.backward()
        self.optimiser.step()

        #Update nn_target

        if t%self.update_target_interval==0:
            print("UPDATING TARGET")
            for target_param, param in zip(self.learner.target_nn.parameters(), self.learner.prediction_nn.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            return 1

    def step(self, time_target:int):

        updates = 0

        while len(self.environment.metro.all_stations)<=10:

            print("iteration_coord:", len(self.environment.metro.all_stations))

            self.time+=1
            L = self.feed_play(self.total_suggerable, self.allowed_per_play)
            #print("LOSS", L)
            self.epsilon*=self.epsilon_decay

            updating = self.backprop(L, self.n_iter*time_target+self.time)
            if updating is not None:
                updates+=1

        return updates

    def reset(self):

        self.average_reward /= self.total_reward
        out = self.average_reward

        self.average_reward=0
        self.total_reward=0
        self.time=0

        self.environment.reset()

        return out
    
    def display(self, show=True):

        frame = self.environment.metropolis.display()
        plt.imshow(frame, cmap='viridis', origin='lower')

        LINES = self.environment.metro.display(True)
        print(LINES)

        # Predefined list of colors.
        colors = ["red", "darkorange", "gold", "yellow", "lime", "green", "cyan","dodgerblue", "blue","purple","blueviolet", "magenta",
                   "pink", "crimson", "maroon"]

        for idx, (key, line) in enumerate(LINES.items()):
            color = colors[idx % len(colors)]  # Cycle through colors if there are more lines than colors.
            for i in range(len(line)-1):
                x_values = [line[i][0], line[i+1][0]]
                y_values = [line[i][1], line[i+1][1]]
                plt.plot(x_values, y_values, color, label=f"Line {key}" if i == 0 else "", marker="o")

        plt.colorbar(label='Density')
        plt.title('Density and Points')

        plt.savefig('result_map.png')
        plt.legend()

        if show:
            plt.show()
        else:
            plt.close()
    

###############################


metro_params={

    #Distances
    "speed_metro" : 10,
    "speed_change" : 1,
    "speed_walk" : 1,

    #Times
    "waiting_for_train": 2,
    "waiting_when_stopping": 0.3,

    "max_connected" : 2, # A change station has at most 2 connections (CANNOT BE 0)

    "r_walking" : 5,
    "k_walking" : 2,
    "make_connection_distance":3,

    "p_selecting_station":0.87 # chance of prolongating a line instead of randomly selecting a station
}

city_params={
    "p_center" : 0.3,
    "p_new" : 0.2,
    "p_station": 0.2,
    "at_most_new":3
}

learning_var={
    "epsilon":0.9,
    "epsilon_decay":0.9999,
    "tau":0.001,
    "update_target_interval":10,
    "gamma":0.98

}

allowed_per_play = 7
total_suggerable = 20

coord = Coordinator(50, 500, (0,0), city_params, (0,0), metro_params, 20000, learning_var, 12, allowed_per_play , total_suggerable)


all = []
target_updates = []
target_updates_r = []
time_target=0

for i in range(2000):

    print("reset", i)
    updating = coord.step(time_target)
    time_target+=1 #Keep track for the update of the target

    if i%10==0:
        print("DISPLAYING")
        coord.display(show=False)

    r = coord.reset()
    print("r:", r)
    print("epsilon:", coord.epsilon)
    #print(coord.stations_network.all_stations)
    all.append(r)

    if updating>0:
        target_updates.append(time_target-1)
        target_updates_r.append(r)

    # Generate your plot with the current state of your_list
    plt.figure()
    plt.plot(all)
    plt.scatter(target_updates, target_updates_r, color='red')
    plt.title('Rewards')
    
    # Save the figure to the same file location every time
    plt.savefig('reward.png')
    
    # Close the figure to free memory
    plt.close()

#Last one to plot:
coord.step(time_target)


print("all_rewards:", all)

coord.display()