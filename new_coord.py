from env import ReplayBuffer, Environment
from learner import Learner

import numpy as np
import torch 
from collections import deque
import random
import matplotlib.pyplot as plt
import time
import copy

def euclidean(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.sqrt(np.sum((a - b) ** 2, axis=-1))


def calculate_averages(lst, chunk_size=100):
    return [np.mean(lst[i:i + chunk_size]) for i in range(0, len(lst), chunk_size)]

class Coordinator:

    def __init__(self, size, n_simulations, city_center, city_facts, first_station, metro_facts, buffer_size, learning_var, n_iter, total_suggerable):

        self.size = size

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
        self.final_reward = 0
        self.total_reward = 0
        self.time = 0

        self.n_iter = n_iter

        self.total_suggerable = total_suggerable

        self.previous_r=0


    def feed_play(self, final):
        
        L = None

        #Select the station that has the best Q_value or at random (using epsilon proba)
        #At random is more likely to select stations at start or end of lines (see env.select_station)
        
        #T.append((time.time()-start_time, "beg"))
        if np.random.uniform(0,1)>self.epsilon:
            all_Q = {}
            with torch.no_grad():
                for station in self.environment.metro.all_stations:

                    if station not in self.environment.metro.complete or station.previous is None or station.next is None:

                        state = self.environment.make_state(station, self.n_iter, final, 0)
                        Q_values = self.learner.prediction_nn.forward(state)
                        all_Q[station] = Q_values[:,1:]

            if all_Q!={}: #Need this in case all stations are in complete (unlikely but happens)
                averages = [(key, torch.max(value).item()) for key, value in all_Q.items()]
                best_station = max(averages, key=lambda x: x[1])[0]
            else:
                best_station = self.environment.select_station()
        
        else:
            best_station = self.environment.select_station()

        #T.append((time.time()-start_time, "select q"))

        #Using selected station, make state and predict
        state = self.environment.make_state(best_station, self.n_iter, final, 0)
        #T.append((time.time()-start_time, "make state"))
        self.learner.predict(state, self.epsilon, best_station.location, self.environment)
        #T.append((time.time()-start_time, "predict"))
        action = self.learner.action

        #Change metro and city with selected action
        new = self.environment.change_metro(best_station, action)
        #self.environment.change_metropolis()

        #Get reward
        #T.append((time.time()-start_time, "change metro"))

        if new is None:
            r = 0
        #elif new==0:
        #r = max(self.environment.get_reward(),0)/2 
        else:
            r = self.environment.get_reward()
            #new_r = self.environment.get_reward()
            #r = 5*max((new_r-self.previous_r),0)
            #self.previous_r=copy.deepcopy(new_r)

        #Make new state 
        new_state = self.environment.make_state(best_station, self.n_iter, final, 1)
        #T.append((time.time()-start_time, "make state 2"))

        self.buffer.push((state, action, r, new_state, final))

        self.learner.target(new_state, r, final)

        #T.append((time.time()-start_time, "push to replay"))

        self.average_reward+=r
        self.total_reward+=1

        if final is True:
            self.final_reward = max(self.environment.get_reward() ,0)
        
        ###############
        #Now sample from REPLAY BUFFER:
        samples = self.buffer.sample(128)

        for sample in samples:

            self.learner.predict_for_replay(sample[0], sample[1])
            self.learner.target(sample[3], sample[2], sample[4])

            if L is None:
                L = self.learner.get_loss()

            else:
                L += self.learner.get_loss()
        #T2.append(time.time()-start_time)

        #print("TIME", T, T2)

        return L/(len(samples)) #Need to divide
    
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

        #while len(self.environment.metro.all_stations)<self.n_iter:

        final=False
        for i in range(self.n_iter):

            print("iteration_coord:", i, len(self.environment.metro.all_stations))

            if i==self.n_iter-1:
                final=True

            self.time+=1
            L = self.feed_play(final)
            #print("got new", len(self.environment.metro.all_stations))
            #print("LOSS", L)
            self.epsilon*=self.epsilon_decay

            updating = self.backprop(L, self.n_iter*time_target+self.time)
            if updating is not None:
                updates+=1

        return updates

    def reset(self):

        self.average_reward /= self.total_reward
        out = self.average_reward
        final_r = self.final_reward

        self.previous_r=0
        self.average_reward=0
        self.total_reward=0
        self.final_reward=0
        self.time=0

        self.environment.reset()

        return out, final_r
    
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

        plt.savefig('result_map'+str(run_number)+'.png')
        plt.legend()

        if show:
            plt.show()
        else:
            plt.close()

        if len(self.environment.metro.all_stations)>=2:

            i_initial, j_initial = self.environment.metro.all_stations[0].location
            i_final, j_final = self.environment.metro.all_stations[-1].location

            initial = (i_initial, j_initial)
            final = (i_final, j_final)

            ok = self.environment.metro.get_fastest(initial, final, display=True, run_number=run_number)

        else:
            print("ALERT")
            for i in self.environment.metro.all_stations:
                print(i.location)

    

###############################


metro_params={

    #Distances
    "speed_metro" : 15,
    "speed_change" : 1,
    "speed_walk" : 1,

    #Times
    "waiting_for_train": 1, #2
    "waiting_when_stopping": 0.6, #0.3

    "max_connected" : 2, # A change station has at most 2 connections (CANNOT BE 0)

    "r_walking" : 15,
    "k_walking" : 6,
    "make_connection_distance":3,

    "p_selecting_station":0.8 # chance of prolongating a line instead of randomly selecting a station (when not picking station based on best q_value)
}

city_params={
    "p_center" : 0.3,
    "p_new" : 0.2,
    "p_station": 0.2,
    "at_most_new":3
}

learning_var={
    "epsilon":0.85,
    "epsilon_decay":0.99995,
    "tau":0.05,
    "update_target_interval":60,
    "gamma":0.98

}


total_suggerable = 20

n_iter = 10

run_number = 1

start_time = time.time()
coord = Coordinator(35, 200, (0,0), city_params, (0,0), metro_params, 300000, learning_var, n_iter , total_suggerable)


all = []
all_final = []
target_updates = []
target_updates_r = []
time_target=0

for i in range(80000):

    print("reset", i)
    updating = coord.step(time_target)
    time_target+=1 #Keep track for the update of the target

    if i%20==0:
        print("DISPLAYING")
        coord.display(show=False)

    r, final_r = coord.reset()
    print("r:", r)
    print("epsilon:", coord.epsilon)
    #print(coord.stations_network.all_stations)
    all.append(r)
    all_final.append(final_r)

    if updating>0:
        target_updates.append(time_target-1)
        target_updates_r.append(r)

    # Generate your plot with the current state of your_list
        
    averages = calculate_averages(all)
    averages_final = calculate_averages(all_final)
    plt.figure()
    plt.plot(averages)
    plt.plot(averages_final)
    #plt.scatter(target_updates, target_updates_r, color='red')
    plt.title('Rewards')
    
    # Save the figure to the same file location every time
    plt.savefig('reward_'+str(run_number)+'.png')
    
    # Close the figure to free memory
    plt.close()

#Last one to plot:
coord.step(time_target)


print("all_rewards:", all)

coord.display()

