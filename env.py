from city_env import Metropolis, City
from metro_env import Station, Line, Stations_network

import numpy as np
import torch 
from collections import deque
import random
import copy

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

        self.waiting_for_train = metro_facts["waiting_for_train"]
        self.waiting_when_stopping = metro_facts["waiting_when_stopping"]

        self.r_walking = metro_facts["r_walking"]
        self.k_walking = metro_facts["k_walking"]
        self.make_connection_distance = metro_facts["make_connection_distance"]

        self.p_selecting_station = metro_facts["p_selecting_station"]

        self.metropolis = Metropolis(self.city_center, self.size, self.p_center, self.p_new, self.p_station)
        self.metro = Stations_network(self.size, self.first_station, self.max_connected, self.speed_walk, self.speed_metro, self.speed_change, self.r_walking, self.k_walking, self.waiting_for_train, self.waiting_when_stopping, self.make_connection_distance)

        for _ in range(10):
            self.metropolis.step(self.at_most_new)

        self.info = None

        self.before_reward = 0
    
    ################################################ 1.
    def make_state(self, selected):

        neighbours = [(1,0), (0,1), (-1,0), (0, -1), (np.sqrt(2)/2, np.sqrt(2)/2), (-np.sqrt(2)/2, -np.sqrt(2)/2), (np.sqrt(2)/2, -np.sqrt(2)/2), (-np.sqrt(2)/2, np.sqrt(2)/2)]
        scales = [8, 15]

        best_dens, max_dens, max_area = self.metropolis.get_best_city_centers(5)

        self.info=torch.zeros((1,90))

        J=0

        #Info about best cities:
        for city in best_dens:
            self.info[0, J] = city[0]/(self.size)+0.5
            self.info[0, J+1] = city[1]/(self.size)+0.5
            self.info[0, J+2] = best_dens[city][0]/max_dens
            self.info[0, J+3] = best_dens[city][1]/max_area
            J+=4

        ##Info about the station itself:
        self.info[0, J]=selected.location[0]/(self.size)+0.5
        self.info[0, J+1]=selected.location[1]/(self.size)+0.5

        dens, area = self.get_dense_around(selected.location)
        dens_occupied  = self.get_share_already_served(selected.location)

        self.info[0, J+2]=dens/area
        self.info[0, J+3]=dens_occupied/dens

        if selected.previous is None or selected.next is None: 
            self.info[0, J+4]=1

        if selected in self.metro.complete:
            self.info[0, J+5]=1

        J+=6

        #Info about possible locations:

        for scale in range(len(scales)):
            for neighb in range(len(neighbours)):

                possible = (selected.location[0]+int(scale*neighb[0]), selected.location[1]+int(scale*neighb[1]))

                self.info[0, J]=possible[0]/(self.size)+0.5
                self.info[0, J+1]=possible[1]/(self.size)+0.5

                dens, area = self.get_dense_around(possible)
                dens_occupied  = self.get_share_already_served(possible)

                self.info[0, J+2]=dens/area
                self.info[0, J+3]=dens_occupied/dens

                #closest, closest_dis = self.get_nearest_station(possible)

                #self.info[0, J+4] = closest.location[0]/(self.size)+0.5
                #self.info[0, J+5] = closest.location[1]/(self.size)+0.5
                #self.info[0, J+6] = closest_dis/np.max(scales)

                J+=4

        return self.info

        """

        best_dens, max_dens, max_area = self.metropolis.get_best_city_centers(5)

        self.info=torch.zeros((1,80+6*self.max_connected))

        #Normalised position
        self.info[0, 0]=selected.location[0]/(self.size)+0.5
        self.info[0, 1]=selected.location[1]/(self.size)+0.5

        if selected.location in self.metropolis.area:
            self.info[0, 2]=self.metropolis.density[selected.location][0]/max_dens


        #If in central city or not
        if selected.location in self.metropolis.central_city.area:
            self.info[0, 3]=1

        #If it is complete, we cannot add any more line
        if selected in self.metro.complete:
            self.info[0, 4]=1

        self.info[0,5]=len(self.metro.all_stations)/10

        j=0

        #Add the densities at the desired points
        for neighb in neighbours:
            for n in scales:

                possible = (selected.location[0]+int(n*neighb[0]), selected.location[1]+int(n*neighb[1]))
                #print("possible", possible)
                if possible[0]<self.size/2 and possible[0]>-self.size/2 and possible[1]<self.size/2 and possible[1]>-self.size/2:
                    if possible in self.metropolis.area:
                        #print("possible", possible, self.metropolis.density[possible][0])
                        self.info[0, 6+j]=self.metropolis.density[possible][0]/max_dens
                        self.info[0, 7+j]=self.metro.station_already(possible)[0]
                        self.info[0, 8+j]=self.metro.station_already(possible)[1]
                j+=3

        J = 6+j
        for city in best_dens:
            self.info[0, J] = city[0]/(self.size)+0.5
            self.info[0, J+1] = city[1]/(self.size)+0.5
            self.info[0, J+2] = best_dens[city][0]/max_dens
            self.info[0, J+3] = best_dens[city][1]/max_area
            J+=4

        #Look at the neighbours of the station

        if selected.previous is not None:
            self.info[0, J]=selected.previous.location[0]/(self.size)+0.5
            self.info[0, J+1]=selected.previous.location[1]/(self.size)+0.5

        else:
            self.info[0, J+2]=1
        
        if selected.next is not None:
            self.info[0, J+3]=selected.next.location[0]/(self.size)+0.5
            self.info[0, J+4]=selected.next.location[1]/(self.size)+0.5

        else:
            self.info[0, J+5]=1

        J+=6
        #Look at connections of the station
        n=0

        if selected.connected!={}:
            for co in selected.connected:

                if co.previous is not None:
                    self.info[0, J+n*6]=co.previous.location[0]/(self.size)+0.5
                    self.info[0, J+n*6+1]=co.previous.location[1]/(self.size)+0.5
                
                else:
                    self.info[0, J+n*6+2]=1
                
                if co.next is not None:
                    self.info[0, J+n*6+3]=co.next.location[0]/(self.size)+0.5
                    self.info[0, J+n*6+4]=co.next.location[1]/(self.size)+0.5

                else:
                    self.info[0, J+n*6+5]=1

                n+=1

        return self.info

    """
    
    ################################################ 2.
    def change_metro(self, station, action):

        neighbours = [(1,0), (0,1), (-1,0), (0, -1), (np.sqrt(2)/2, np.sqrt(2)/2), (-np.sqrt(2)/2, -np.sqrt(2)/2), (np.sqrt(2)/2, -np.sqrt(2)/2), (-np.sqrt(2)/2, np.sqrt(2)/2)]
        scales = [8, 15]

        if action!=0:
            n = scales[(action-1)//8]
            direction = neighbours[(action-1)%8]

            #New location added
            new_loc = (station.location[0] + int(n*direction[0]), station.location[1] + int(n*direction[1]))

            if new_loc[0]<self.size/2 and new_loc[0]>-self.size/2 and new_loc[1]<self.size/2 and new_loc[1]>-self.size/2:
                new = self.metro.make_new_station(station, new_loc[0], new_loc[1], returning=True)

                if new is not None:
                    self.metro.make_connection_close(new)

    
    ################################################ 3.
    def change_metropolis(self):

        self.metropolis.step(self.at_most_new)

        for station in self.metro.all_stations:
            if random.uniform(0,1)<self.p_station: #Don't grow stations everytime

                self.metropolis.step_station(station.location)


    ################################################ 4.
    def get_reward(self):


        reward=0
        _, max_dens, _ = self.metropolis.get_best_city_centers(5)


        for _ in range(self.n_simulations):

            i_initial, j_initial = self.metropolis.pick_point()
            i_final, j_final = self.metropolis.pick_point()

            while euclidean((i_initial, j_initial),(i_final, j_final))<5:
                i_initial, j_initial = self.metropolis.pick_point()

            initial = (i_initial, j_initial)
            final = (i_final, j_final)

            walking_time, metro_time, _ = self.metro.get_fastest(initial, final)

            if metro_time==np.inf:
                reward += 0

            else:
                x = (metro_time-walking_time)/walking_time
                if x<=1/4:
                    reward+= -(4/9)*x+1/9

                
        return reward/self.n_simulations

    
    ################################################ 5.
    def reset(self):

        self.metropolis = Metropolis(self.city_center, self.size, self.p_center, self.p_new, self.p_station)
        self.metro = Stations_network(self.size, self.first_station, self.max_connected, self.speed_walk, self.speed_metro, self.speed_change, self.r_walking, self.k_walking, self.waiting_for_train, self.waiting_when_stopping, self.make_connection_distance)

        for _ in range(10):
            self.metropolis.step(self.at_most_new)

    ################################################ 6.
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
        
    ################################################ 7.
    def get_dense_around(self, location):

        density=0
        total_seen = 0
        for pixel in self.metropolis.all_possible_pixels:

            if euclidean(pixel, location)<=self.metro.r_walking:

                if pixel in self.metropolis.area:
                    density+=self.metropolis.density[pixel][0]

                total_seen+=1

        return density, total_seen
    
    def get_nearest_station(self, location):

        closest = None
        dis_closest = np.inf
        for station in self.metro.all_stations:
            if euclidean(station.location, location)<dis_closest:
                closest = station
                dis_closest = euclidean(station.location, location)

        return closest, dis_closest
    
    def get_share_already_served(self, location):

        density_served = 0
        for pixel in self.metropolis.area:
            if euclidean(pixel, location)<=self.metro.r_walking:

                for station in self.metro.all_stations:
                    if euclidean(station.location, pixel)<=self.metro.r_walking:
                        density_served+= self.metropolis.density[pixel][0]

        return density_served
                



        


