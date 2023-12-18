from city_env import Metropolis, City
from metro_env import Station, Line, Stations_network

import numpy as np
import torch 
from collections import deque
import random

def euclidean(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.sqrt(np.sum((a - b) ** 2, axis=-1))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience):
        # Adds an experience to the buffer. Oldest experiences are automatically dropped if capacity is reached.
        self.buffer.append(experience)

    def sample(self, batch_size):
        # Randomly samples a batch of experiences from the buffer.
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)



class Environment():

    def __init__ (self, size, n_simulations, city_center, city_facts, first_station, metro_facts):

        self.size = size
        self.n_simulations = n_simulations

        self.city_center = city_center
        self.p_center = city_facts["p_center"]
        self.p_new = city_facts["p_new"]
        self.p_station = city_facts["p_station"]
        self.at_most_new = city_facts["at_most_new"]

        self.first_station = first_station
        self.max_connected = metro_facts["max_connected"]
        self.speed_walk = metro_facts["speed_walk"]
        self.speed_metro = metro_facts["speed_metro"]
        self.speed_change = metro_facts["speed_change"]
        self.r_walking = metro_facts["r_walking"]
        self.k_walking = metro_facts["k_walking"]

        self.p_selecting_station = metro_facts["p_selecting_station"]

        self.metropolis = Metropolis(self.city_center, self.size, self.p_center, self.p_new, self.p_station)
        self.metro = Stations_network(self.size, self.first_station, self.max_connected, self.speed_walk, self.speed_metro, self.speed_change, self.r_walking, self.k_walking)

        for _ in range(10):
            self.metropolis.step(self.at_most_new)

        self.info = None
    
    ################################################ 1.
    def make_state(self, selected):

        neighbours = [(1,0), (0,1), (-1,0), (0, -1), (np.sqrt(2)/2, np.sqrt(2)/2), (-np.sqrt(2)/2, -np.sqrt(2)/2), (np.sqrt(2)/2, -np.sqrt(2)/2), (-np.sqrt(2)/2, np.sqrt(2)/2)]
        scales = [5, 15, 25, 50]

        self.info=torch.zeros((1, 38+2*self.max_connected))
        i=0

        #Normalised position
        self.info[i, 0]=selected.location[0]/(self.size)+0.5
        self.info[i, 1]=selected.location[1]/(self.size)+0.5

        #If in central city or not
        if selected.location in self.metropolis.central_city.area:
            self.info[i, 2]=1

        j=0

        #Add the densities at the desired points
        for neighb in neighbours:
            for n in scales:

                possible = (selected.location[0]+int(n*neighb[0]), selected.location[1]+int(n*neighb[1]))
                
                if possible[0]<self.size/2 and possible[0]>-self.size/2 and possible[1]<self.size/2 and possible[1]>-self.size/2:
                    if possible in list(self.metropolis.density.keys()):
                        self.info[i, 3+j]=self.metropolis.density[possible][0]
                j+=1

        #Look at the neighbours of the station
        J = 3+j
        if selected.previous is not None:
            self.info[i, J]=1/100*(selected.location[0]-selected.previous.location[0])+0.5
        
        if selected.next is not None:
            self.info[i, J+1]=1/100*(selected.location[1]-selected.next.location[1])+0.5

        if selected.previous is None and selected.next is None:
            self.info[i, J+2]=1

        J+=3
        #Look at connections of the station
        if selected.connected!={}:
            n=0
            for co in selected.connected:

                if co.previous is not None:
                    self.info[i, J+n*self.max_connected]=1/100*(co.location[0]-co.previous.location[0])+0.5
                
                if co.next is not None:
                    self.info[i, J+n*self.max_connected+1]=1/100*(co.location[1]-co.next.location[1])+0.5

                n+=1

        return self.info
    
    ################################################ 2.
    def change_metro(self, station, action):

        neighbours = [(1,0), (0,1), (-1,0), (0, -1), (np.sqrt(2)/2, np.sqrt(2)/2), (-np.sqrt(2)/2, -np.sqrt(2)/2), (np.sqrt(2)/2, -np.sqrt(2)/2), (-np.sqrt(2)/2, np.sqrt(2)/2)]
        scales = [10, 25, 50]

        if action!=0:
            n = scales[(action-1)//8]
            direction = neighbours[(action-1)%8]

            #New location added
            self.metro.make_new_station(station, int(n*direction[0]), int(n*direction[1]))

    
    ################################################ 2.
    def change_metropolis(self):

        self.metropolis.step(self.at_most_new)

        for station in self.metro.all_stations:
            if random.uniform(0,1)<self.p_station: #Don't grow stations everytime

                self.metropolis.step_station(station.location)


    ################################################ 3.
    def get_reward(self):

        reward=0

        for _ in range(self.n_simulations):

            i_initial, j_initial = self.metropolis.pick_point()
            i_final, j_final = self.metropolis.pick_point()

            while euclidean((i_initial, j_initial),(i_final, j_final))<5:
                i_initial, j_initial = self.metropolis.pick_point()

            initial = (i_initial, j_initial)
            final = (i_final, j_final)

            walking_time, metro_time, _ = self.metro.get_fastest(initial, final)


            if 8*metro_time<walking_time:
                reward+=1.2
            elif 7.5*metro_time<walking_time:
                reward+=1
            elif 7*metro_time<walking_time:
                reward+=0.85
            elif 6*metro_time<walking_time:
                reward+=0.66
            elif 5*metro_time<walking_time:
                reward+=0.2
            elif 4*metro_time<walking_time:
                reward-=0.2
            elif 3*metro_time<walking_time:
                reward-=0.5
            elif 2*metro_time<walking_time:
                reward-=0.7
            elif metro_time<walking_time:
                reward-=1
            elif 0.5*metro_time<walking_time:
                reward-=1.2
            else:
                reward-=2
                
        return reward/self.n_simulations
    
    ################################################ 4.
    def reset(self):

        self.metropolis = Metropolis(self.city_center, self.size, self.p_center, self.p_new, self.p_station)
        self.metro = Stations_network(self.size, self.first_station, self.max_connected, self.speed_walk, self.speed_metro, self.speed_change, self.r_walking, self.k_walking)

        for _ in range(10):
            self.metropolis.step(self.at_most_new)

    ################################################ 5.
    def select_station(self):

        if np.random.uniform(0,1)<self.p_selecting_station:

            line = np.random.randint(1, len(list(self.metro.lines.keys()))+1)
            station_ind = np.random.randint(0, 2)

            if station_ind==0:
                station = self.metro.lines[line].starting

            else:
                station = self.metro.lines[line].ending

            return station

        else:
            return self.metro.all_stations[np.random.randint(0, len(self.metro.all_stations))]


        


